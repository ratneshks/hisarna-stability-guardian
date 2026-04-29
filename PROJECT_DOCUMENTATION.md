# HIsarna Stability Guardian: Project Documentation

## 1. Executive Summary
The **HIsarna Stability Guardian** is a professional-grade industrial AI demonstration system developed for **Tata Steel R&D**. It utilizes a **Physics-Informed Neural Network (PINN)** to monitor the stability of the HIsarna ironmaking reactor in real-time. Unlike traditional neural networks that rely solely on data, this system embeds the fundamental laws of physics (fluid dynamics and heat transfer) directly into its learning process, allowing it to predict instabilities before they occur.

---

## 2. The Industrial Problem
The HIsarna process is a revolutionary ironmaking technology using a **Cyclone Converter Furnace (CCF)**. It is highly efficient but faces several operational challenges:
*   **Non-Linear Dynamics:** The furnace operates at ~1450°C with simultaneous fluid dynamics, combustion heat transfer, and chemical gas reactions.
*   **Instability Triggers:** Injection of high-alumina ore often causes unexplained pressure and temperature fluctuations.
*   **PID Limitations:** Traditional controllers (PID) cannot handle the multi-physics coupling of the system.
*   **Safety Risks:** Rapid instability can lead to "thermal runaway" or tuyere blockages, necessitating immediate operator intervention.

---

## 3. The Solution: Physics-Informed Neural Networks (PINNs)
Standard AI models often make "unphysical" predictions (e.g., negative oxygen levels). This project solves that by using a **PINN architecture**.
*   **Physics Constraints:** We embed the **Navier-Stokes** (fluid flow) and **Convective-Diffusive Heat** equations into the AI's loss function.
*   **Hybrid Learning:** The model learns from historical sensor data *while* ensuring its predictions respect the laws of physics.
*   **Explainability:** By monitoring "Physics Residuals," we can tell if the reactor is deviating from its physical equilibrium, providing a higher level of trust than a "black-box" model.

---

## 4. System Architecture
The project is organized as a unified monorepo:
*   `/data`: A high-fidelity synthetic generator that simulates 10 critical furnace sensors.
*   `/model`: The core AI engine containing the PyTorch PINN, the preprocessing logic, and the stability envelope.
*   `/backend`: A FastAPI server that handles real-time inference and streams data via WebSockets.
*   `/frontend`: A React-based industrial dashboard for real-time monitoring and operator alerts.

---

## 5. Module Deep Dive

### A. Synthetic Data Generation (`/data/generator.py`)
Since real-world iROC sensor data is proprietary, we built a simulator that produces 1 Hz data across 10 channels:
1.  **Temperature:** Cyclone and Gas Outlet temps.
2.  **Pressure:** Cyclone and Differential pressures.
3.  **Flow:** Gas velocity and Ore injection rates.
4.  **Composition:** Oxygen ($O_2$), $CO$, and $CO_2$ fractions.
5.  **Vibration:** RMS amplitude (early warning indicator).

The generator supports three modes: **Stable**, **Warning** (ore injection spikes), and **Critical** (O2 depletion/thermal runaway).

### B. Preprocessing & Feature Engineering (`/model/preprocessing.py`)
The system doesn't just look at a single data point; it analyzes a **30-second sliding window**. 
*   **Normalization:** Uses Z-score scaling based on a "stable" baseline.
*   **Derived Features:** Computes rates of change ($\Delta T$), linear trends in $O_2$, and combustion efficiency indices.

### C. The PINN Model (`/model/pinn.py`)
A 4-layer deep neural network using **Tanh activation functions** (necessary for smooth physics derivatives).
*   **Custom Loss Function:**
    *   **Data Loss:** Ensures predictions match sensor readings.
    *   **Navier-Stokes Loss:** Ensures gas flow predictions follow fluid dynamics laws.
    *   **Heat Loss:** Ensures temperature predictions follow thermodynamic laws.
    *   **Boundary Loss:** Prevents "unphysical" states (e.g., negative O2).

### D. Stability Envelope (`/model/stability.py`)
We use the **Mahalanobis Distance** algorithm on the hidden "latent" layers of the neural network. This creates a multi-dimensional "safety bubble" around stable operation. When the reactor's state moves outside this bubble, the system classifies it as a Warning or Critical event.

### E. Real-Time Backend (`/backend/main.py`)
*   **WebSocket Engine:** Pushes updates every 1 second to the dashboard.
*   **Alert Engine:** Monitors for threshold breaches (e.g., $T > 1520^\circ C$) and generates prescriptive recommendations for the operator.

### F. Industrial Dashboard (`/frontend/src/`)
A high-performance monitoring UI featuring:
*   **Live KPI Cards:** Instant status of stability and physics compliance.
*   **Time-Series Trends:** Historical view of sensor data vs. AI predictions.
*   **Prescriptive Alerts:** Tells the operator exactly what to do (e.g., "Reduce ore injection by 15%").

---

## 6. How the System Operates
1.  **Initialization:** The system loads a pre-trained PINN model.
2.  **Ingestion:** Real-time sensor data is fed into a 30s buffer.
3.  **Inference:** The AI predicts the next state and checks for physics violations.
4.  **Classification:** The Mahalanobis distance determines if the process is "Stable."
5.  **Alerting:** If instability is detected, the UI pulses red/yellow and issues an instruction.

---

## 7. Future Scale
For a full industrial rollout at Tata Steel, this system would:
1.  Connect directly to the **iROC (Integrated Remote Operation Centre)** data stream.
2.  Integrate with the **Level 2 Automation** to provide closed-loop control.
3.  Be retrained on real historical "degradation" events to further sharpen its predictive accuracy.
