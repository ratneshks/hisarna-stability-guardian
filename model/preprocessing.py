import numpy as np
import pandas as pd
import torch

class HIsarnaPreprocessor:
    def __init__(self):
        self.mu_dict = {}
        self.sigma_dict = {}
        self.channels = [
            'T_cyclone', 'T_gas_outlet', 'P_cyclone', 'P_differential',
            'v_gas', 'phi_O2', 'phi_CO', 'phi_CO2', 'm_ore', 'vibration_rms'
        ]
        
    def fit(self, df_warmup):
        """Fit the z-score normalizer on a warmup dataframe of stable data"""
        for col in self.channels:
            if col in df_warmup.columns:
                self.mu_dict[col] = df_warmup[col].mean()
                self.sigma_dict[col] = df_warmup[col].std()
                if self.sigma_dict[col] == 0:
                    self.sigma_dict[col] = 1e-6 # Prevent div by zero

    def normalize(self, df):
        """Apply z-score normalization"""
        df_norm = pd.DataFrame(index=df.index)
        for col in self.channels:
            if col in df.columns and col in self.mu_dict:
                df_norm[col] = (df[col] - self.mu_dict[col]) / self.sigma_dict[col]
            else:
                df_norm[col] = 0.0
        return df_norm

    def extract_features(self, df):
        """Extract the 5 derived features from a raw (unnormalized) 30s window dataframe"""
        features = {}
        
        # delta_T over last 10s
        if len(df) >= 10:
            features['delta_T'] = (df['T_cyclone'].iloc[-1] - df['T_cyclone'].iloc[-10]) / 10.0
        else:
            features['delta_T'] = 0.0
            
        # phi_O2_trend (linear slope) over last 30s
        if len(df) > 1:
            y = df['phi_O2'].values
            x = np.arange(len(y))
            slope, _ = np.polyfit(x, y, 1)
            features['phi_O2_trend'] = slope
        else:
            features['phi_O2_trend'] = 0.0
            
        # P_ratio (current)
        curr = df.iloc[-1]
        features['P_ratio'] = curr['P_differential'] / curr['P_cyclone']
        
        # combustion_index
        denom = (curr['phi_CO'] + curr['phi_CO2'])
        features['combustion_index'] = curr['phi_CO'] / denom if denom > 0 else 0
        
        # ore_T_interaction (using normalized values ideally, but here raw mapped is ok)
        m_ore_norm = (curr['m_ore'] - self.mu_dict.get('m_ore', 1.1)) / self.sigma_dict.get('m_ore', 1.0)
        T_cyc_norm = (curr['T_cyclone'] - self.mu_dict.get('T_cyclone', 1450.0)) / self.sigma_dict.get('T_cyclone', 1.0)
        features['ore_T_interaction'] = m_ore_norm * T_cyc_norm
        
        return features

    def process_window(self, df_window, current_time=0.0):
        """
        Process a 30s window dataframe into a 308-dim torch Tensor for the PINN.
        df_window must be exactly 30 rows.
        """
        if len(df_window) < 30:
            # Pad if necessary (mostly for startup)
            pad_size = 30 - len(df_window)
            pad_df = pd.DataFrame([df_window.iloc[0]] * pad_size)
            df_window = pd.concat([pad_df, df_window]).reset_index(drop=True)
        elif len(df_window) > 30:
            df_window = df_window.iloc[-30:].reset_index(drop=True)
            
        # Extract features from RAW data
        features = self.extract_features(df_window)
        feature_vec = [
            features['delta_T'],
            features['phi_O2_trend'],
            features['P_ratio'],
            features['combustion_index'],
            features['ore_T_interaction']
        ]
        
        # Normalize window
        df_norm = self.normalize(df_window)
        window_flattened = df_norm.values.flatten().tolist() # 30 * 10 = 300
        
        # Coordinates
        x_coord = 0.5
        y_coord = 0.7
        t_coord = (current_time % 3600.0) / 3600.0
        
        final_input = window_flattened + feature_vec + [x_coord, y_coord, t_coord]
        
        tensor_input = torch.tensor(final_input, dtype=torch.float32)
        # require_grad is needed for PDE residual computation on coordinates if we want, 
        # but typically the coordinates inside the input are part of the graph.
        tensor_input.requires_grad_(True)
        return tensor_input
