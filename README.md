# Analytics Application

A full-stack analytics application with FastAPI backend and Next.js frontend for CSV data processing and intelligent querying.

## 📚 Additional Resources

### Google Colab Notebooks
- **[Analytics Notebook 1](https://colab.research.google.com/drive/14__QVUJ4U4V0Q9brqQ9HcONf9gBxB2qL?usp=sharing)** - Data processing and analysis workflows
- **[Analytics Notebook 2](https://colab.research.google.com/drive/1rKSrKG_ZOGOD-OGzS-9N1TqUhmAsFeUc?usp=sharing)** - Advanced analytics and visualization

### Product Demo
- **[📹 Product Demo Video](https://drive.google.com/drive/folders/1bLQFI_WP-CTLlJuGljIox8eaXwjK0psh?usp=drive_link)** - Complete walkthrough of the analytics application features

## 🚀 Quick Start

### macOS/Linux
```bash
./start.sh
```

### Windows
```cmd
start.bat
```

## ✨ What the startup script does

The startup script automatically handles the complete setup and launch process:

1. **Backend Setup** (Port 8001):
   - Creates/recreates Python virtual environment with correct paths
   - Installs all required dependencies from `requirements.txt`
   - Starts FastAPI server with auto-reload using `uvicorn main:app --host 0.0.0.0 --port 8001 --reload`

2. **Frontend Setup** (Port 3001):
   - Installs Node.js dependencies if `node_modules` doesn't exist
   - Starts Next.js development server on port 3001 using `npm run dev`

3. **Process Management**:
   - Runs both services concurrently in the background
   - Provides graceful shutdown with `Ctrl+C`
   - Automatically cleans up all processes on exit

## 🌐 Access Points

Once running, you can access:

- **📱 Frontend Application**: http://localhost:3001
- **🔧 Backend API**: http://localhost:8001  
- **📚 API Documentation**: http://localhost:8001/docs
- **🩺 Health Check**: http://localhost:8001/health

## 🔧 Manual Setup

If you prefer to run the services manually:

### Backend
```bash
cd analytics-backend
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate.bat
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### Frontend
```bash
cd analytics-frontend
npm install
PORT=3001 npm run dev
```

## 🛑 Stopping the Application

- **macOS/Linux**: Press `Ctrl+C` in the terminal running the script
- **Windows**: Press any key when prompted in the command window

The script will automatically clean up all running processes and provide confirmation.

## 📊 Features

- **CSV Processing**: Upload and process multiple CSV files
- **Data Quality Analysis**: Comprehensive data quality scoring
- **Intelligent Querying**: Natural language queries on your data
- **Demographics Analysis**: State-city distribution and compliance reporting
- **Geographic Analysis**: Data quality metrics by state
- **Export Functionality**: Download cleaned datasets

## 🔧 API Endpoints

- `POST /process` - Process CSV files and run data quality analysis
- `GET /demographics` - Get state-city distribution and compliance status
- `GET /geographics` - Get data quality metrics by state
- `GET /expiry` - Generate expired license compliance report
- `POST /chat` - Intelligent query processing with natural language
- `GET /export` - Export cleaned roster CSV file

## 📋 Requirements

- **Backend**: Python 3.8+, FastAPI, uvicorn
- **Frontend**: Node.js 18+, Next.js 14+, React 18+
- **System**: macOS, Linux, or Windows
