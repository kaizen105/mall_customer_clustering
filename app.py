import os
import json
from flask import Flask, request, jsonify
import joblib
import numpy as np

# --config--
MODEL_PATH = os.getenv('MODEL_PATH', 'model/customer_segmentation.pkl')
SCALER_PATH = os.getenv('SCALER_PATH', 'model/scaler.pkl')

# --App
app = Flask(__name__)

# load model & scaler at startup
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model or scaler: {e}")

@app.get('/health')
def health():
    return {"status": "ok"}, 200

@app.post('/predict')
def predict():
    '''
    accepts either:
    {"input": [[income, spending], [income, spending], ...]} #2D list
    or
    {"input": [income, spending]} #1D list
    '''
    try:
        payload = request.get_json(force=True)
        x = payload.get('input')
        if x is None:
            return jsonify(error="Missing 'input'"), 400

        # normalize to 2D array
        if isinstance(x, list) and (len(x) > 0) and isinstance(x[0], list):
            X = x
        else:
            X = [x]

        X = np.array(X, dtype=float)

        # scale features before prediction
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)

        return jsonify(predictions=preds.tolist()), 200
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
