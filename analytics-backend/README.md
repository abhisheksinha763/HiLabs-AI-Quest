# CSV Analytics Backend

A FastAPI backend service for processing CSV files uploaded from a Next.js frontend. This service provides basic data analysis and statistics for uploaded CSV files.

## Features

- **Multi-file CSV Upload**: Process multiple CSV files simultaneously
- **Data Analysis**: Basic statistics, column analysis, and data profiling
- **Error Handling**: Robust error handling for malformed files and encoding issues
- **CORS Support**: Configured for Next.js frontend integration
- **Health Checks**: Built-in health check endpoints

## API Endpoints

### `GET /`
Root endpoint that returns service status.

### `GET /health`
Health check endpoint for monitoring service availability.

### `POST /process`
Main endpoint for processing CSV files.

**Request**: Multipart form data with CSV files
**Response**: JSON with analysis results for each file

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd analytics-backend
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Server

#### Development Mode
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Production Mode
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

#### Using Python directly
```bash
python main.py
```

The server will be available at:
- **Local**: http://localhost:8000
- **Network**: http://0.0.0.0:8000
- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Alternative Docs**: http://localhost:8000/redoc

## Configuration

### CORS Settings
The backend is configured to accept requests from:
- `http://localhost:3000` (Next.js development)
- `https://your-site.com` (production - update as needed)

To modify CORS settings, edit the `allow_origins` list in `main.py`.

### File Upload Limits
- **Supported formats**: CSV files (`.csv`, `text/csv`, `application/vnd.ms-excel`)
- **File size**: No explicit limit set (FastAPI default is 16MB)
- **Multiple files**: Supported

## API Response Format

### Success Response
```json
{
  "status": "success",
  "message": "Successfully processed 2 file(s)",
  "files": [
    {
      "filename": "data.csv",
      "file_size_bytes": 1024,
      "rows": 100,
      "cols": 5,
      "columns": ["id", "name", "age", "city", "salary"],
      "numeric_columns": ["id", "age", "salary"],
      "text_columns": ["name", "city"],
      "numeric_stats": {
        "age": {
          "mean": 35.5,
          "median": 34.0,
          "min": 18,
          "max": 65,
          "null_count": 0
        }
      },
      "missing_values_total": 5,
      "duplicate_rows": 2,
      "sample_data": [...]
    }
  ]
}
```

### Error Response
```json
{
  "detail": "Unsupported file type: application/pdf. Only CSV files are allowed."
}
```

## Integration with Next.js Frontend

### Environment Variables
In your Next.js project, add to `.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Example Frontend Code
```typescript
const handleUpload = async (files: FileList) => {
  const formData = new FormData();
  Array.from(files).forEach(file => formData.append('files', file));

  const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/process`, {
    method: 'POST',
    body: formData,
  });

  const result = await response.json();
  console.log(result);
};
```

## Development

### Project Structure
```
analytics-backend/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### Adding New Features
1. Add new endpoints in `main.py`
2. Update dependencies in `requirements.txt` if needed
3. Test with the automatic API documentation at `/docs`

## Deployment

### Docker (Optional)
Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Considerations
- Use a production ASGI server like Gunicorn with Uvicorn workers
- Set up proper logging and monitoring
- Configure environment variables for different stages
- Set up SSL/TLS certificates for HTTPS
- Configure proper CORS origins for your production domain

## Troubleshooting

### Common Issues

1. **CORS Errors**: Update the `allow_origins` list in `main.py`
2. **File Upload Errors**: Check file format and size
3. **Encoding Issues**: The backend handles common encoding problems automatically
4. **Port Conflicts**: Change the port in the uvicorn command

### Logs
The server logs will show detailed information about requests and any errors that occur.

## License

This project is open source and available under the MIT License.
