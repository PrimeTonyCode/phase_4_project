from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
PIPELINE_PATH = os.path.join(BASE_DIR, "random_forest_model.joblib") 

model = joblib.load(PIPELINE_PATH)

# Define possible classes
class_names = model.classes_

# FastAPI app
app = FastAPI()

# Request body schema
class TweetRequest(BaseModel):
    tweets: list[str]  # Expecting a list of raw text tweets


@app.get("/")
def read_root():
    return {'message': 'Twitter Sentiments Model API'}

@app.post("/predict")
def predict(request: TweetRequest):
    try:

        # Build DataFrame with correct column name as used in notebook
        df = pd.DataFrame({"tweet_text": request.tweets})

        # Predict
        predictions = model.predict(df)

        # If predictions are strings, use them directly
        if isinstance(predictions[0], str):
            mapped_predictions = predictions
        else:
            # If predictions are encoded as integers, map to string labels
            # Try to get mapping from model.classes_ if available
            try:
                mapped_predictions = [model.classes_[pred] for pred in predictions]
            except Exception:
                # Fallback to hardcoded mapping
                label_map = {0: "negative", 1: "positive", 2: "neutral"}
                mapped_predictions = [label_map.get(pred, "unknown") for pred in predictions]

        # Return each tweet alongside its prediction
        results = [
            {"tweet": tweet, "sentiment": sentiment}
            for tweet, sentiment in zip(request.tweets, mapped_predictions)
        ]

        return {"predictions": results}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})