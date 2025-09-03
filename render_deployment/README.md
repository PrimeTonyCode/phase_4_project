# BERT Sentiment Analysis API

A Flask API for sentiment analysis using a fine-tuned BERT model.

## Deployment Steps on Render

1. Create a new Web Service on Render:
   - Go to https://dashboard.render.com/
   - Click "New +" and select "Web Service"
   - Connect your GitHub repository

2. Configure the Web Service:
   - Name: `bert-sentiment-analysis` (or your preferred name)
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Select the appropriate instance type (at least 512MB RAM recommended)

3. Add Environment Variables (if needed):
   - `PORT`: 5000
   - Any other configuration variables your app needs

## API Usage

### POST /predict
Analyzes the sentiment of provided text.

Request:
```json
{
    "text": "Your tweet text here"
}
```

Response:
```json
{
    "text": "Your tweet text here",
    "sentiment": "positive",
    "confidence": "95.32%",
    "probabilities": {
        "negative": "2.45%",
        "neutral": "2.23%",
        "positive": "95.32%"
    }
}
```

### GET /health
Health check endpoint.

## Local Testing

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run the Flask app:
```bash
python app.py
```

3. Test with curl:
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text":"This is an amazing product!"}'
```
