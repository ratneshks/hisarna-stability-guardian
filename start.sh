#!/bin/bash
echo "Starting HIsarna Stability Guardian..."

# Setup Python Backend
echo "Setting up Python environment..."
python -m venv venv
source venv/Scripts/activate || source venv/bin/activate
pip install -r requirements.txt

# Check if model exists, train if not
if [ ! -f "model/checkpoints/hisarna_pinn.pt" ]; then
    echo "Model checkpoint not found. Training in demo mode..."
    python model/train.py --demo
fi

# Start FastAPI server in background
echo "Starting FastAPI backend..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Setup Frontend
echo "Setting up React frontend..."
cd frontend
npm install

# Start Vite dev server
echo "Starting Vite dev server..."
npm run dev &
FRONTEND_PID=$!

echo "HIsarna Stability Guardian is running. Open http://localhost:5173"

# Wait for background processes
wait $BACKEND_PID
wait $FRONTEND_PID
