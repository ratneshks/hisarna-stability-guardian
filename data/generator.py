import time
import math
import random
import numpy as np
import pandas as pd
from collections import deque

class HIsarnaDataGenerator:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.mode = "stable"
        self.time_step = 0
        self.history = deque(maxlen=3600)  # Keep up to 1 hour of history
        self.current_reading = {}
        
        # Base states
        self.state = {
            'T_cyclone': 1450.0,
            'T_gas_outlet': 940.0,
            'P_cyclone': 102000.0,
            'P_differential': 10000.0,
            'v_gas': 23.0,
            'phi_O2': 4.0,
            'phi_CO': 30.0,
            'phi_CO2': 25.0,
            'm_ore': 1.1,
            'vibration_rms': 0.05
        }
        
        # To handle slow transitions
        self.target_state = self.state.copy()
        self.transition_rate = 0.05 # How fast state moves to target state
        
        self._generate_step() # Initial reading

    def _generate_step(self):
        self.time_step += 1
        t = self.time_step
        
        # Mode management (random transitions if not forced)
        # Normally handled externally for scenarios, but here's base logic:
        # In a real background generator, we'd use probabilities. 
        # For the demo, we rely mostly on forced scenarios.
        
        # Determine targets based on mode
        if self.mode == "stable":
            # Slow sinusoidal drift
            self.target_state['T_cyclone'] = 1450.0 + 15.0 * math.sin(2 * math.pi * t / 120)
            self.target_state['P_cyclone'] = 102000.0 + 500.0 * math.cos(2 * math.pi * t / 120)
            self.target_state['m_ore'] = 1.1 + 0.1 * math.sin(2 * math.pi * t / 60)
            self.target_state['phi_O2'] = 4.0
            self.target_state['v_gas'] = 23.0
            self.target_state['vibration_rms'] = 0.05
            self.target_state['P_differential'] = 10000.0
            self.transition_rate = 0.1
            
        elif self.mode == "warning":
            self.target_state['m_ore'] = 1.4
            self.target_state['phi_O2'] = 1.5
            self.target_state['T_cyclone'] = 1495.0
            self.target_state['v_gas'] = 27.6 # 20% increase
            self.target_state['vibration_rms'] = 0.1
            self.target_state['P_differential'] = 11000.0
            self.transition_rate = 0.02 # Gradual onset (45s)
            
        elif self.mode == "critical":
            self.target_state['m_ore'] = 1.4
            self.target_state['phi_O2'] = 0.3
            self.target_state['T_cyclone'] = 1540.0
            self.target_state['P_differential'] = 8000.0 # >15% drop
            self.target_state['vibration_rms'] = 0.25
            self.transition_rate = 0.1 # Abrupt onset (<10s)

        # Apply smooth transitions towards targets
        for k in self.state.keys():
            self.state[k] += (self.target_state[k] - self.state[k]) * self.transition_rate
            
        # Add correlated / physical constraints
        self.state['T_gas_outlet'] = 0.85 * self.state['T_cyclone'] - 292.5 + np.random.normal(0, 5)
        self.state['phi_CO'] = 30.0 + (4.0 - self.state['phi_O2']) * 1.5 + np.random.normal(0, 0.5)
        self.state['phi_CO2'] = 60.0 - self.state['phi_CO'] - self.state['phi_O2'] + np.random.normal(0, 0.5)
        
        # Add specific noise
        self.current_reading = {
            'T_cyclone': self.state['T_cyclone'] + np.random.normal(0, 5),
            'T_gas_outlet': self.state['T_gas_outlet'],
            'P_cyclone': self.state['P_cyclone'] + np.random.normal(0, 150),
            'P_differential': self.state['P_differential'] + np.random.normal(0, 100),
            'v_gas': self.state['v_gas'] + np.random.normal(0, 1.5),
            'phi_O2': max(0.01, self.state['phi_O2'] + np.random.normal(0, 0.2)),
            'phi_CO': max(0.1, self.state['phi_CO']),
            'phi_CO2': max(0.1, self.state['phi_CO2']),
            'm_ore': max(0.1, self.state['m_ore'] + np.random.normal(0, 0.05)),
            'vibration_rms': max(0.01, self.state['vibration_rms'] + np.random.normal(0, 0.01)),
            'timestamp': time.time()
        }
        
        self.history.append(self.current_reading.copy())
        
    def get_latest_reading(self):
        return {k: v for k, v in self.current_reading.items() if k != 'timestamp'}

    def get_history(self, n_seconds):
        n_seconds = min(n_seconds, len(self.history))
        data = list(self.history)[-n_seconds:]
        return pd.DataFrame(data)

    def set_mode(self, mode):
        if mode in ["stable", "warning", "critical"]:
            self.mode = mode
