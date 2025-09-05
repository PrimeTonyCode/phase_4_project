#!/bin/bash

# Script to run the FastAPI application locally for testing

# Check if python is installed
if ! command -v python &> /dev/null; then
    echo "Python is not installed. Please install Python first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "pip is not installed. Please install pip first."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r bert_requirements.txt

# Run the FastAPI application
echo "Starting FastAPI server..."
uvicorn app_bert:app --reload

echo "Server is running at http://localhost:8000"
