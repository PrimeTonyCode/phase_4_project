# Run the FastAPI application locally for testing

# Check if python is installed
if (Get-Command python -ErrorAction SilentlyContinue) {
    # Python is installed
    Write-Host "Found Python installation"
} else {
    Write-Host "Python is not installed. Please install Python first."
    exit 1
}

# Install dependencies
Write-Host "Installing dependencies..."
pip install -r bert_requirements.txt

# Run the FastAPI application
Write-Host "Starting FastAPI server..."
python -m uvicorn app_bert:app --reload

Write-Host "Server is running at http://localhost:8000"
