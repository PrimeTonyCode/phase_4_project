from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import os
import joblib
import numpy as np

# Define base directory
BASE_DIR = os.path.dirname(__file__)

# Load Random Forest model and vectorizer
MODEL_PATH = os.path.join(BASE_DIR, "bert_deployment", "random_forest_model.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "bert_deployment", "text_vectorizer.joblib")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# FastAPI app
app = FastAPI()

# Request body schema
class TweetRequest(BaseModel):
    tweets: list[str]  # Expecting a list of raw text tweets

@app.get("/")
def read_root():
    return {'message': 'Twitter Sentiments Random Forest API'}

@app.post("/predict")
def predict(request: TweetRequest):
    try:
        results = []
        sentiment_labels = ['negative', 'neutral', 'positive']
        X = vectorizer.transform(request.tweets)
        preds = model.predict(X)
        probs = model.predict_proba(X)
        for i, text in enumerate(request.tweets):
            predicted_sentiment = sentiment_labels[preds[i]]
            confidence_scores = {
                label: f"{probs[i][j]*100:.2f}%" 
                for j, label in enumerate(sentiment_labels)
            }
            results.append({
                "tweet": text,
                "sentiment": predicted_sentiment,
                "confidence": confidence_scores[predicted_sentiment],
                "probabilities": confidence_scores
            })
        return {"predictions": results}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
def health():
    return {"status": "healthy", "model": "RandomForest"}
