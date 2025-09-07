@echo off
setlocal enabledelayedexpansion

REM Analytics Application Startup Script for Windows
REM Starts both backend (FastAPI) and frontend (Next.js) applications

echo 🚀 Starting Analytics Application...
echo ==================================

REM Check if required directories exist
if not exist "analytics-backend" (
    echo ❌ Error: analytics-backend directory not found
    pause
    exit /b 1
)

if not exist "analytics-frontend" (
    echo ❌ Error: analytics-frontend directory not found
    pause
    exit /b 1
)

REM Start Backend (FastAPI on port 8001)
echo 🔧 Starting backend on port 8001...
cd analytics-backend

REM Check if virtual environment exists
if not exist ".venv" (
    echo ⚠️  Virtual environment not found. Creating one...
    python -m venv .venv
)

REM Activate virtual environment and install dependencies
call .venv\Scripts\activate.bat
pip install -r requirements.txt >nul 2>&1

REM Start backend server in background
start /b python -c "import uvicorn; from main import app; uvicorn.run(app, host='0.0.0.0', port=8001, reload=False)"
echo ✅ Backend started

REM Go back to root directory
cd ..

REM Start Frontend (Next.js on port 3001)
echo 🔧 Starting frontend on port 3001...
cd analytics-frontend

REM Install dependencies if node_modules doesn't exist
if not exist "node_modules" (
    echo 📦 Installing frontend dependencies...
    npm install
)

REM Start frontend server in background
start /b cmd /c "set PORT=3001 && npm run dev"
echo ✅ Frontend started

REM Go back to root directory
cd ..

echo.
echo 🎉 Analytics Application is now running!
echo ==================================
echo 📊 Backend API:  http://localhost:8001
echo 🌐 Frontend:     http://localhost:3001
echo 📋 API Docs:     http://localhost:8001/docs
echo.
echo Press any key to stop all services...
pause >nul

REM Cleanup - kill processes
echo.
echo 🛑 Shutting down applications...
taskkill /f /im python.exe >nul 2>&1
taskkill /f /im node.exe >nul 2>&1
echo ✅ Applications stopped successfully
pause
