#!/bin/bash

# Analytics Application Startup Script
# Starts both backend (FastAPI) and frontend (Next.js) applications

set -e

echo "🚀 Starting Analytics Application..."
echo "=================================="

# Function to cleanup processes on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down applications..."
    
    # Kill background processes
    if [ ! -z "$BACKEND_PID" ]; then
        echo "Stopping backend (PID: $BACKEND_PID)..."
        kill $BACKEND_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$FRONTEND_PID" ]; then
        echo "Stopping frontend (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    
    echo "✅ Applications stopped successfully"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check if required directories exist
if [ ! -d "analytics-backend" ]; then
    echo "❌ Error: analytics-backend directory not found"
    exit 1
fi

if [ ! -d "analytics-frontend" ]; then
    echo "❌ Error: analytics-frontend directory not found"
    exit 1
fi

# Start Backend (FastAPI on port 8001)
echo "🔧 Starting backend on port 8001..."
cd analytics-backend

# Check if virtual environment exists, if not create a new one
if [ ! -d ".venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv .venv
fi

# Remove any existing virtual environment that might have wrong paths
if [ -f ".venv/pyvenv.cfg" ]; then
    # Check if the venv path is correct
    if ! grep -q "$(pwd)" .venv/pyvenv.cfg; then
        echo "🔄 Recreating virtual environment with correct path..."
        rm -rf .venv
        python3 -m venv .venv
    fi
fi

# Activate virtual environment and install dependencies
source .venv/bin/activate
echo "📦 Installing backend dependencies..."
pip install -r requirements.txt

# Start backend server
uvicorn main:app --host 0.0.0.0 --port 8001 --reload &

BACKEND_PID=$!
echo "✅ Backend started (PID: $BACKEND_PID)"

# Go back to root directory
cd ..

# Start Frontend (Next.js on port 3001)
echo "🔧 Starting frontend on port 3001..."
cd analytics-frontend

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi

# Start frontend server
PORT=3001 npm run dev &
FRONTEND_PID=$!
echo "✅ Frontend started (PID: $FRONTEND_PID)"

# Go back to root directory
cd ..

echo ""
echo "🎉 Analytics Application is now running!"
echo "=================================="
echo "📊 Backend API:  http://localhost:8001"
echo "🌐 Frontend:     http://localhost:3001"
echo "📋 API Docs:     http://localhost:8001/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for processes to complete
wait $BACKEND_PID $FRONTEND_PID
