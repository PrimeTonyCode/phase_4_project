#!/bin/bash

echo "Testing BERT model API..."

# Check if curl is installed
if ! command -v curl &> /dev/null; then
    echo "Error: curl is not installed. Please install curl first."
    exit 1
fi

# URL to test (default to localhost if no argument provided)
URL=${1:-"http://localhost:8000"}

# Test health endpoint
echo -e "\nTesting health endpoint..."
curl -s "$URL/health"

# Test predict endpoint
echo -e "\n\nTesting predict endpoint..."
curl -s -X POST \
  "$URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tweets": [
      "I love this product! It works amazingly well.",
      "This is the worst purchase I have ever made."
    ]
  }'

echo -e "\n\nTest completed!"
