# BERT Model Sentiment Analysis API

This is a Flask API that serves the BERT model for sentiment analysis of tweets.

## API Endpoints

### POST /predict
Analyzes the sentiment of provided text using the BERT model.

**Request Body:**
```json
{
    "text": "Your tweet text here"
}
```

**Response:**
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

## Deployment Steps

1. Ensure your BERT model is saved in the `bert-saved` directory

2. Install Heroku CLI and log in:
   ```
   heroku login
   ```

3. Create a new Heroku app:
   ```
   heroku create your-app-name
   ```

4. Set up Git:
   ```
   git init
   git add .
   git commit -m "Initial deployment"
   ```

5. Configure Heroku buildpacks:
   ```
   heroku buildpacks:set heroku/python
   ```

6. Push to Heroku:
   ```
   git push heroku master
   ```

7. Ensure the app is running:
   ```
   heroku ps:scale web=1
   ```

## Local Testing

1. Install requirements:
   ```
   pip install -r requirements.txt
   ```

2. Run the Flask app:
   ```
   python app.py
   ```

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
