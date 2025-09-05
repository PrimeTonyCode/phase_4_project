from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Define base directory
BASE_DIR = os.path.dirname(__file__)

# Load the model and tokenizer - adjust path if needed
MODEL_PATH = os.path.join(BASE_DIR, "bert_deployment")
try:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback to remote model if local files are missing
    try:
        print("Attempting to load from remote model...")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        model.eval()
    except Exception as e:
        print(f"Error loading fallback model: {e}")

# FastAPI app
app = FastAPI()

# Request body schema
class TweetRequest(BaseModel):
    tweets: list[str]  # Expecting a list of raw text tweets

@app.get("/")
def read_root():
    return {'message': 'Twitter Sentiments BERT API'}

@app.post("/predict")
def predict(request: TweetRequest):
    try:
        results = []
        
        for text in request.tweets:
            # Tokenize input
            inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
            
            # Make prediction
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get predicted class and probabilities
            predicted_class = torch.argmax(predictions, dim=1).item()
            probabilities = predictions[0].tolist()
            
            # Convert labels
            sentiment_labels = ['negative', 'neutral', 'positive']
            predicted_sentiment = sentiment_labels[predicted_class]
            
            # Create confidence scores
            confidence_scores = {
                label: f"{prob*100:.2f}%" 
                for label, prob in zip(sentiment_labels, probabilities)
            }
            
            # Add result
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
    return {"status": "healthy", "model": "BERT"}
