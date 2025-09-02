from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('../random_forest_model.joblib')
vectorizer = joblib.load('../text_vectorizer.joblib')

def predict_sentiment(text, model=None, vectorizer=None):
    """
    Predict sentiment of a given text using the trained Random Forest model.
    """
    # Convert single text to list if needed
    if isinstance(text, str):
        text = [text]
    
    # Vectorize the text
    X = vectorizer.transform(text)
    
    # Get predictions and probabilities
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Create results
    results = []
    for pred, prob in zip(predictions, probabilities):
        confidence = max(prob) * 100
        result = {
            'text': text[len(results)],
            'sentiment': pred,
            'confidence': f"{confidence:.2f}%",
            'probabilities': {
                label: f"{p*100:.2f}%" 
                for label, p in zip(model.classes_, prob)
            }
        }
        results.append(result)
    
    return results[0] if len(text) == 1 else results

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the text from the JSON request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        # Make prediction
        result = predict_sentiment(text, model, vectorizer)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
