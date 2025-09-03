from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json

def load_model_and_tokenizer(model_path):
    """Load the saved model and tokenizer"""
    print(f"Loading model and tokenizer from {model_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()  # Set to evaluation mode
    return model, tokenizer

def predict_sentiment(text, model, tokenizer):
    """Make prediction using the loaded model"""
    # Tokenize the input text
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Get predicted class and probabilities
    predicted_class = torch.argmax(predictions, dim=1).item()
    probabilities = predictions[0].tolist()
    
    # Convert numerical labels to sentiment labels (adjust based on your model's classes)
    sentiment_labels = ['negative', 'neutral', 'positive']
    predicted_sentiment = sentiment_labels[predicted_class]
    
    # Create confidence scores
    confidence_scores = {
        label: f"{prob*100:.2f}%" 
        for label, prob in zip(sentiment_labels, probabilities)
    }
    
    return {
        'text': text,
        'sentiment': predicted_sentiment,
        'confidence': confidence_scores[predicted_sentiment],
        'probabilities': confidence_scores
    }

def main():
    # Path to saved model
    model_path = "./bert-saved"
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # Test cases
    test_texts = [
        "This product is absolutely amazing! Best purchase ever!",
        "Terrible service, would not recommend to anyone.",
        "The product is okay, nothing special but works as expected.",
        "Really impressed with the quality and customer service!",
        "Disappointed with the delivery time and packaging."
    ]
    
    print("\nRunning test predictions...")
    print("-" * 50)
    
    # Make predictions
    for text in test_texts:
        result = predict_sentiment(text, model, tokenizer)
        print(f"\nInput: {result['text']}")
        print(f"Predicted Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']}")
        print("Probability Distribution:")
        for sentiment, prob in result['probabilities'].items():
            print(f"  {sentiment}: {prob}")
        print("-" * 50)

if __name__ == "__main__":
    main()
