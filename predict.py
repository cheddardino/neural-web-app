"""
api/predict.py
Dress Code Compliance — SVM Inference Endpoint
"""

from flask import Flask, request, jsonify
import os, io, pickle
import numpy as np
from PIL import Image
from skimage.feature import hog

app = Flask(__name__)

# ── Model loading (cached after first cold start) ──────────────────────────
_model = None
_scaler = None

def get_models():
    global _model, _scaler
    if _model is None:
        base = os.path.join(os.path.dirname(__file__), '..', 'models')
        with open(os.path.join(base, 'svm_model.pkl'), 'rb') as f:
            _model = pickle.load(f)
        with open(os.path.join(base, 'scaler.pkl'), 'rb') as f:
            _scaler = pickle.load(f)
    return _model, _scaler


def preprocess(file_stream):
    """Resize → RGB → HOG (matches Part 1 & 2 pipeline exactly)."""
    img = Image.open(file_stream).convert('RGB').resize((128, 128))
    arr = np.array(img)
    features = hog(
        arr,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        channel_axis=-1
    )
    return features


# ── CORS helper ────────────────────────────────────────────────────────────
def _cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return _cors(jsonify({}))

    try:
        model, scaler = get_models()

        if 'image' not in request.files:
            return _cors(jsonify({'error': 'No image file provided.'})), 400

        features = preprocess(request.files['image'].stream)
        features_scaled = scaler.transform([features])
        pred = int(model.predict(features_scaled)[0])

        # Confidence — use predict_proba if available, else decision_function
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features_scaled)[0]
            confidence = round(float(max(probs)) * 100, 1)
        else:
            df = float(model.decision_function(features_scaled)[0])
            confidence = round(min(abs(df) * 15 + 55, 99.0), 1)

        return _cors(jsonify({
            'label': 'Compliant' if pred == 1 else 'Non-Compliant',
            'confidence': confidence,
            'compliant': bool(pred == 1)
        }))

    except Exception as e:
        return _cors(jsonify({'error': str(e)})), 500
