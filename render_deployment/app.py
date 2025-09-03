from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

app = Flask(__name__)

# Load the model and tokenizer
model_path = './bert_deployment'
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

def predict_sentiment(text):
    """Predict sentiment for given text"""
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
    
    return {
        'text': text,
        'sentiment': predicted_sentiment,
        'confidence': confidence_scores[predicted_sentiment],
        'probabilities': confidence_scores
    }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        result = predict_sentiment(text)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'BERT'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
