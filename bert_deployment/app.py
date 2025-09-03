from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

app = Flask(__name__)

# Load the model and tokenizer
model_path = '../bert-saved'
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set the model to evaluation mode
model.eval()

def predict_sentiment(text):
    """
    Predict sentiment using the BERT model
    """
    try:
        # Tokenize the input text
        inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get predicted class and probabilities
        predicted_class = torch.argmax(predictions, dim=1).item()
        probabilities = predictions[0].tolist()
        
        # Convert numerical labels to sentiment labels (adjust these based on your model's classes)
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
    
    except Exception as e:
        return {'error': str(e)}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        result = predict_sentiment(text)
        
        if 'error' in result:
            return jsonify(result), 500
            
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'BERT'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
