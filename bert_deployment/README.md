# Twitter Sentiment Analysis BERT API

This repository contains a FastAPI application that serves a BERT model for sentiment analysis on tweets.

## API Endpoints

### POST /predict
Analyzes the sentiment of provided tweets using the BERT model.

**Request Body:**
```json
{
    "tweets": [
        "I love this new iPhone! The camera quality is amazing.",
        "This Android phone keeps crashing. Terrible experience."
    ]
}
```

**Response:**
        {
            "tweet": "I love this new iPhone! The camera quality is amazing.",
            "sentiment": "positive",
            "confidence": "98.76%",
            "probabilities": {
                "negative": "0.12%",
                "neutral": "1.12%",
                "positive": "98.76%"
            }
        },
        {
            "tweet": "This Android phone keeps crashing. Terrible experience.",
            "sentiment": "negative",
            "confidence": "95.34%",
            "probabilities": {
                "negative": "95.34%",
                "neutral": "4.21%",
                "positive": "0.45%"
            }
        }
    ]
}
```

### GET /health
Health check endpoint.

## Local Setup

1. Install dependencies:
   ```
   pip install -r bert_requirements.txt
   ```

2. Run the API locally:
   ```
   python -m uvicorn app_bert:app --reload
   ```

   Or use the provided scripts:
   - For Windows: `.\run_local.ps1`
   - For Linux/Mac: `bash run_local.sh`

   The API will be available at http://localhost:8000

## Deployment on Render

This repository is configured for deployment on Render using Docker:

1. Push your code to GitHub
2. Connect your GitHub repository to Render
3. Create a new Web Service and select "Deploy from GitHub"
4. Select the repository and use the following settings:
   - Environment: Docker
   - Docker file path: Dockerfile
   - Region: Oregon (or your preferred region)

The API will be deployed and available at the URL provided by Render.

## Directory Structure

- `app_bert.py` - FastAPI application
- `bert_requirements.txt` - Required Python packages
- `Dockerfile` - Docker configuration for Render deployment
- `render.yaml` - Render configuration file
- `run_local.ps1` - Windows script to run locally
- `run_local.sh` - Linux/Mac script to run locally
- `model.safetensors` - BERT model weights
- Other model files (tokenizer, config, etc.)

3. Test with curl:
   ```
   curl -X POST http://localhost:5000/predict \
        -H "Content-Type: application/json" \
        -d '{"text":"This is an amazing product!"}'
   ```

## Notes
- The API uses the BERT model saved in the `bert-saved` directory
- Make sure you have enough memory on your deployment instance as BERT models can be resource-intensive
- Consider using Heroku's performance dynos for better response times
