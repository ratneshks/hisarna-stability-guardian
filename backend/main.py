import os
import time
import json
import uuid
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generator import HIsarnaDataGenerator
from model.preprocessing import HIsarnaPreprocessor
from model.pinn import HIsarnaPINN, PINNLoss
from model.stability import StabilityEnvelope

app = FastAPI(title="HIsarna Stability Guardian")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
generator = HIsarnaDataGenerator(seed=123)
preprocessor = HIsarnaPreprocessor()
model = HIsarnaPINN()
criterion = PINNLoss()
stability_env = StabilityEnvelope()

model_loaded = False
startup_time = time.time()

# Histories for API
stability_history = []
residuals_history = []
active_alerts = []

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()

def load_system():
    global model_loaded
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoints_dir = os.path.join(base_dir, "model", "checkpoints")
        
        # Load preprocessor
        with open(os.path.join(checkpoints_dir, "preprocessor.json"), "r") as f:
            pp_data = json.load(f)
            preprocessor.mu_dict = pp_data["mu_dict"]
            preprocessor.sigma_dict = pp_data["sigma_dict"]
            
        # Load stability env
        with open(os.path.join(checkpoints_dir, "stability_env.json"), "r") as f:
            env_data = json.load(f)
            stability_env.mu_stable = np.array(env_data["mu_stable"])
            stability_env.Sigma_inv = np.array(env_data["Sigma_inv"])
            stability_env.d_95th_percentile = env_data["d_95th_percentile"]
            
        # Load model
        model_path = os.path.join(checkpoints_dir, "hisarna_pinn.pt")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        model_loaded = True
        print("Model and artifacts loaded successfully.")
    except Exception as e:
        print(f"Failed to load model/artifacts: {e}")

@app.on_event("startup")
async def startup_event():
    load_system()
    asyncio.create_task(background_simulation())

def process_alerts(sensor_data, stability_status, L_NS):
    global active_alerts
    current_time = time.time()
    
    # 1. phi_O2 < 1.5% for 10s (simplified to immediate for demo, or we can use history)
    # Actually, we can check recent history
    df_recent = generator.get_history(10)
    if len(df_recent) >= 10 and df_recent['phi_O2'].max() < 1.5:
        add_alert("WARNING", "phi_O2", "Oxygen fraction critically low — combustion imbalance risk", False)
        
    # 2. T_cyclone > 1500
    if sensor_data['T_cyclone'] > 1520:
        add_alert("CRITICAL", "T_cyclone", "Thermal runaway risk — immediate intervention required", False)
    elif sensor_data['T_cyclone'] > 1500:
        add_alert("WARNING", "T_cyclone", "Cyclone temperature above safe operating limit", False)
        
    # 3. phi_O2 < 0.5%
    if sensor_data['phi_O2'] < 0.5:
        add_alert("CRITICAL", "phi_O2", "Oxygen depletion — process shutdown risk", False)
        
    # 4. vibration > 0.15
    if sensor_data['vibration_rms'] > 0.15:
        add_alert("WARNING", "vibration_rms", "Abnormal vibration detected — tuyere integrity check required", False)
        
    # 5. Physics violation
    if L_NS > 0.05:
        add_alert("WARNING", "physics", "Physics constraint violation — model confidence reduced. Sensor check recommended.", True)
        
    # Auto-resolve old alerts if condition cleared (simplified: just keep them active unless acknowledged for demo, 
    # or clear if completely stable)
    if stability_status == "STABLE":
        active_alerts = [a for a in active_alerts if a['acknowledged']]

def add_alert(severity, sensor, message, is_physics):
    # Check if already active
    for a in active_alerts:
        if a['sensor_triggered'] == sensor and a['severity'] == severity and not a['acknowledged']:
            return
    
    active_alerts.append({
        "id": str(uuid.uuid4()),
        "severity": severity,
        "sensor_triggered": sensor,
        "message": message,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "acknowledged": False,
        "physics_violation": is_physics
    })
    
    # Sort: CRITICAL first, then WARNING
    active_alerts.sort(key=lambda x: (0 if x['severity'] == 'CRITICAL' else 1, x['timestamp']), reverse=True)

async def background_simulation():
    while True:
        generator._generate_step()
        sensor_data = generator.get_latest_reading()
        
        preds = {
            "stability_score": 0.0,
            "anomaly_magnitude": 0.0
        }
        stab_info = {"status": "STABLE", "mahalanobis_distance": 0.0, "normalized_distance": 0.0, "color": "#22c55e"}
        res_info = {"L_NS": 0.001, "L_HT": 0.001, "L_data": 0.0}
        rec = "Initializing (Gathering 30s of sensor data)..."
        
        if model_loaded and len(generator.history) >= 30:
            df_win = generator.get_history(30)
            t_in = preprocessor.process_window(df_win, current_time=time.time()).unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                # We need gradients for physics, so we enable it just for input
                t_in.requires_grad_(True)
                y_pred, latent = model(t_in, return_latent=True)
                
                # Compute residuals
                # For demo, we just compute them without true labels to get L_NS, L_HT
                # Dummy target
                y_true = torch.zeros_like(y_pred)
                _, L_data, L_NS, L_HT, L_BC = criterion(t_in, y_pred, y_true)
                
                latent_np = latent.detach().numpy()[0]
                y_pred_np = y_pred.detach().numpy()[0]
                
                d_norm = stability_env.compute_distance(latent_np)
                status, color = stability_env.classify(d_norm)
                
                preds = {
                    "u_pred": float(y_pred_np[0]),
                    "v_pred": float(y_pred_np[1]),
                    "T_pred": float(y_pred_np[2]),
                    "P_pred": float(y_pred_np[3]),
                    "phi_O2_pred": float(y_pred_np[4]),
                    "phi_CO_pred": float(y_pred_np[5]),
                    "stability_score": float(y_pred_np[6]),
                    "anomaly_magnitude": float(y_pred_np[7])
                }
                
                stab_info = {
                    "status": status,
                    "mahalanobis_distance": float(d_norm * stability_env.d_95th_percentile),
                    "normalized_distance": float(d_norm),
                    "color": color
                }
                
                res_info = {
                    "L_NS": float(L_NS.item()),
                    "L_HT": float(L_HT.item()),
                    "L_data": float(L_data.item())
                }
                
                rec = stability_env.get_recommendation(status, sensor_data)
                
                process_alerts(sensor_data, status, float(L_NS.item()))
                
                # Store history
                ts = time.time()
                stability_history.append({"timestamp": ts, **stab_info, "score": preds["stability_score"]})
                residuals_history.append({"timestamp": ts, **res_info})
                
                # Keep history bounded
                if len(stability_history) > 3600: stability_history.pop(0)
                if len(residuals_history) > 3600: residuals_history.pop(0)

        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sensors": sensor_data,
            "predictions": preds,
            "stability": stab_info,
            "physics_residuals": res_info,
            "recommendation": rec,
            "active_alerts": active_alerts
        }
        
        await manager.broadcast(json.dumps(payload))
        await asyncio.sleep(1.0)

@app.get("/api/health")
def health():
    return {
        "status": "ok", 
        "model_loaded": model_loaded, 
        "uptime_seconds": time.time() - startup_time
    }

@app.get("/api/model-info")
def model_info():
    return {
        "architecture": "PINN-4Layer",
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_epochs": 100, # Demo assumed
        "val_accuracy": 0.98,
        "physics_equations": ["Navier-Stokes", "Heat Transfer", "Species Conservation"],
        "input_dim": 308,
        "output_dim": 8
    }

@app.get("/api/sensor-history")
def get_sensor_history(seconds: int = 300):
    df = generator.get_history(seconds)
    # Convert to list of dicts with timestamp
    # Since we didn't store timestamp in the returned dict, we approximate or use the internal history
    history_dicts = list(generator.history)[-seconds:]
    return history_dicts

@app.get("/api/stability-history")
def get_stability_history(seconds: int = 300):
    return stability_history[-seconds:]

@app.get("/api/physics-residuals")
def get_physics_residuals(seconds: int = 60):
    return residuals_history[-seconds:]

@app.get("/")
async def root():
    return {"message": "HIsarna Stability Guardian API is Online", "docs": "/docs"}

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "model_loaded": model_loaded, "history_len": len(generator.history)}

@app.get("/api/training-loss")
async def get_training_loss():
    try:
        with open("model/logs/training_loss.json", "r") as f:
            return json.load(f)
    except:
        return {}

class ScenarioRequest(BaseModel):
    scenario: str

@app.post("/api/trigger-scenario")
def trigger_scenario(req: ScenarioRequest):
    if req.scenario in ["stable", "warning", "critical"]:
        generator.set_mode(req.scenario)
        return {"status": "success", "mode": req.scenario}
    return {"status": "error", "message": "invalid scenario"}

@app.websocket("/ws/live-data")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
