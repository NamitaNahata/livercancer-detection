from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model and feature names
MODEL_PATH = 'model.pkl'
FEATURES_PATH = 'feature_names.pkl'

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(FEATURES_PATH, 'rb') as f:
    feature_names = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = {}
        for feature in feature_names:
            value = request.form.get(feature)
            if value:
                data[feature] = float(value)
        
        # Convert to DataFrame with correct column order
        input_df = pd.DataFrame([data], columns=feature_names)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Get probability if available
        try:
            proba = model.predict_proba(input_df)[0]
            confidence = max(proba) * 100
        except:
            confidence = 0
        
        result = {
            'prediction': 'Cancer Detected' if prediction == 1 else 'No Cancer',
            'confidence': f'{confidence:.2f}%',
            'prediction_value': int(prediction)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
