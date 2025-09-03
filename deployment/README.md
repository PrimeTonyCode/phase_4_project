# Tweet Sentiment Analysis API

This is a Flask API for tweet sentiment analysis using a trained Random Forest model.

## API Endpoints

### POST /predict
Predicts the sentiment of a given text.

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
    "sentiment": "predicted_sentiment",
    "confidence": "confidence_score%",
    "probabilities": {
        "sentiment1": "probability1%",
        "sentiment2": "probability2%"
    }
}
```

### GET /health
Health check endpoint.

## Deployment Steps

1. Install Heroku CLI and log in:
   ```
   heroku login
   ```

2. Create a new Heroku app:
   ```
   heroku create your-app-name
   ```

3. Move your model files to the deployment directory:
   - Move `random_forest_model.joblib` to the deployment folder
   - Move `text_vectorizer.joblib` to the deployment folder

4. Initialize git and push to Heroku:
   ```
   git init
   git add .
   git commit -m "Initial deployment"
   git push heroku master
   ```

5. Ensure the app is running:
   ```
   heroku ps:scale web=1
   ```

## Local Testing

1. Install requirements:
   ```
   pip install -r requirements.txt
   ```

2. Run the app:
   ```
   python app.py
   ```

3. Test with curl:
   ```
   curl -X POST http://localhost:5000/predict \
        -H "Content-Type: application/json" \
        -d '{"text":"This is a great product!"}'
   ```
