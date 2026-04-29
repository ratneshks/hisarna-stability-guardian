import os
import sys
import json
import argparse
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

# Add root directory to path for cloud deployment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generator import HIsarnaDataGenerator
from model.preprocessing import HIsarnaPreprocessor
from model.pinn import HIsarnaPINN, PINNLoss

class ModelTrainer:
    def __init__(self, is_demo=False):
        self.is_demo = is_demo
        self.epochs = 100 if is_demo else 500
        self.batch_size = 64
        self.device = torch.device("cpu") # For demo purposes
        
        self.generator = HIsarnaDataGenerator(seed=42)
        self.preprocessor = HIsarnaPreprocessor()
        self.model = HIsarnaPINN().to(self.device)
        self.criterion = PINNLoss()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=50, factor=0.5)

        self.loss_history = {
            "epoch": [],
            "L_total": [],
            "L_data": [],
            "L_NS": [],
            "L_HT": [],
            "L_BC": [],
            "val_loss": []
        }

    def generate_data(self):
        print("Generating synthetic data...")
        num_stable = 1600 if self.is_demo else 8000
        num_warning = 300 if self.is_demo else 1500
        num_critical = 100 if self.is_demo else 500
        
        data = []
        
        # Warmup
        self.generator.set_mode("stable")
        for _ in range(300):
            self.generator._generate_step()
            
        warmup_df = self.generator.get_history(300)
        self.preprocessor.fit(warmup_df)
        
        # Helper to generate sequences
        def _gen(mode, count):
            self.generator.set_mode(mode)
            for _ in range(count):
                self.generator._generate_step()
                # Need to wait for 30s window to accumulate
                if len(self.generator.history) >= 30:
                    df_win = self.generator.get_history(30)
                    # Use index as current_time approx
                    t_in = self.preprocessor.process_window(df_win, current_time=float(len(data)))
                    
                    # Target: For simplicity, the output is the current normalized reading 
                    # plus an arbitrary mapping for velocity/stability (since we don't have true vel)
                    # Target array: [u, v, T, P, phi_O2, phi_CO, stability, anomaly]
                    curr = self.generator.get_latest_reading()
                    norm_curr = self.preprocessor.normalize(pd.DataFrame([curr])).iloc[0]
                    
                    # Dummy targets for physics
                    u_target = norm_curr['v_gas'] * 0.7
                    v_target = norm_curr['v_gas'] * 0.3
                    stab = 1.0 if mode == 'stable' else (0.5 if mode == 'warning' else 0.0)
                    anom = 0.0 if mode == 'stable' else (0.5 if mode == 'warning' else 1.0)
                    
                    target = [
                        u_target, v_target, norm_curr['T_cyclone'], norm_curr['P_cyclone'],
                        norm_curr['phi_O2'], norm_curr['phi_CO'], stab, anom
                    ]
                    
                    data.append((t_in.detach().numpy(), np.array(target, dtype=np.float32)))

        _gen("stable", num_stable)
        _gen("warning", num_warning)
        _gen("critical", num_critical)
        
        # Shuffle and split
        np.random.shuffle(data)
        split = int(0.8 * len(data))
        
        X = torch.tensor(np.array([d[0] for d in data]))
        Y = torch.tensor(np.array([d[1] for d in data]))
        
        X.requires_grad_(True)
        
        X_train, Y_train = X[:split], Y[:split]
        X_val, Y_val = X[split:], Y[split:]
        
        train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=self.batch_size)
        
        return train_loader, val_loader

    def train(self):
        train_loader, val_loader = self.generate_data()
        
        print(f"Starting training for {self.epochs} epochs...")
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = total_data = total_NS = total_HT = total_BC = 0.0
            
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                
                # In order to compute gradients w.r.t input, it must require grad
                if not X_batch.requires_grad:
                    X_batch.requires_grad_(True)
                    
                self.optimizer.zero_grad()
                Y_pred = self.model(X_batch)
                
                L_total, L_data, L_NS, L_HT, L_BC = self.criterion(X_batch, Y_pred, Y_batch)
                
                L_total.backward()
                self.optimizer.step()
                
                total_loss += L_total.item()
                total_data += L_data.item()
                total_NS += L_NS.item()
                total_HT += L_HT.item()
                total_BC += L_BC.item()
                
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, Y_batch in val_loader:
                    X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                    Y_pred = self.model(X_batch)
                    val_loss += self.criterion.mse(Y_pred, Y_batch).item()
                    
            val_loss /= len(val_loader)
            self.scheduler.step(val_loss)
            
            # Log
            n = len(train_loader)
            self.loss_history["epoch"].append(epoch)
            self.loss_history["L_total"].append(total_loss / n)
            self.loss_history["L_data"].append(total_data / n)
            self.loss_history["L_NS"].append(total_NS / n)
            self.loss_history["L_HT"].append(total_HT / n)
            self.loss_history["L_BC"].append(total_BC / n)
            self.loss_history["val_loss"].append(val_loss)
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d} | L_tot: {total_loss/n:.4f} | L_dat: {total_data/n:.4f} | L_NS: {total_NS/n:.4f} | val: {val_loss:.4f}")

        self.save_checkpoint()

    def save_checkpoint(self):
        os.makedirs("model/checkpoints", exist_ok=True)
        os.makedirs("model/logs", exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), "model/checkpoints/hisarna_pinn.pt")
        
        # Save logs
        with open("model/logs/training_loss.json", "w") as f:
            json.dump(self.loss_history, f)
        
        # Also need to save preprocessor params and stability envelope for the backend
        # To do this correctly, we fit stability envelope here and save it
        print("Fitting stability envelope...")
        self.model.eval()
        latent_vectors = []
        # Generate 5000 stable samples specifically
        self.generator.set_mode("stable")
        for _ in range(5000 if not self.is_demo else 1000):
            self.generator._generate_step()
            if len(self.generator.history) >= 30:
                df_win = self.generator.get_history(30)
                t_in = self.preprocessor.process_window(df_win, current_time=0.0).unsqueeze(0)
                _, latent = self.model(t_in, return_latent=True)
                latent_vectors.append(latent.detach().numpy()[0])
                
        latent_vectors = np.array(latent_vectors)
        from model.stability import StabilityEnvelope
        env = StabilityEnvelope()
        env.fit(latent_vectors)
        
        env_data = {
            "mu_stable": env.mu_stable.tolist() if env.mu_stable is not None else None,
            "Sigma_inv": env.Sigma_inv.tolist() if env.Sigma_inv is not None else None,
            "d_95th_percentile": env.d_95th_percentile
        }
        with open("model/checkpoints/stability_env.json", "w") as f:
            json.dump(env_data, f)
            
        preproc_data = {
            "mu_dict": self.preprocessor.mu_dict,
            "sigma_dict": self.preprocessor.sigma_dict
        }
        with open("model/checkpoints/preprocessor.json", "w") as f:
            json.dump(preproc_data, f)
            
        print("Model and artifacts saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run fast demo training")
    args = parser.parse_args()
    
    trainer = ModelTrainer(is_demo=args.demo)
    trainer.train()
