from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = "model.safetensors"
TOKENIZER_PATH = "vocab.txt"

tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
model.eval()

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
class RequestModel(BaseModel):
    text: str

        # Build DataFrame with correct column name as used in notebook
        df = pd.DataFrame({"tweet_text": request.tweets})

        # Predict
        predictions = model.predict(df)
def predict(request: RequestModel):
        # If predictions are strings, use them directly
        if isinstance(predictions[0], str):
            mapped_predictions = predictions
        inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred = np.argmax(logits.numpy(), axis=1)
        return int(pred[0])
            # If predictions are encoded as integers, map to string labels
            # Try to get mapping from model.classes_ if available
        sentiment = predict_sentiment(request.text)
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