import os
import pickle
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from flask import Flask, render_template, request, jsonify

# ─── sklearn-unpickle patch ───────────────────────────────────────────────────
import sklearn.compose._column_transformer as _ct
class _RemainderColsList: pass
_ct._RemainderColsList = _RemainderColsList
# ────────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder='templates', static_folder='static')\

import pickle

with open("best_model.pkl", "rb") as f:
    model1 = pickle.load(f)


try:
    response = requests.get(model1)
    response.raise_for_status()
    model = pickle.load(BytesIO(response.content))
    FEATURE_NAMES = list(model.feature_names_in_)
    print(f"Loaded model from URL; expecting columns: {FEATURE_NAMES}")
except Exception as e:
    print(f"Error loading model from {model1}:", e)
    model = None
    FEATURE_NAMES = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check the server logs.'})

    try:
        # Get and validate input
        data = {}
        missing = []
        for col in FEATURE_NAMES:
            val = request.form.get(col)
            if val is None:
                missing.append(col)
            else:
                data[col] = val

        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400

        # Convert numeric fields
        for num_col in ['bedroom_number', 'bathroom_number', 'living_space',
                        'land_space', 'price_per_unit', 'postcode']:
            if num_col in data:
                data[num_col] = float(data[num_col])

        # Create single-row DataFrame
        df = pd.DataFrame([data], columns=FEATURE_NAMES)

        # Make prediction
        scaled_price = model.predict(df)[0]
        raw_price = scaled_price * 10000  # Rescale back

        return jsonify({
            'price': round(raw_price, 2),
            'formatted_price': "${:,.2f}".format(round(raw_price, 2))
        })

    except Exception as e:
        return jsonify({'error': f"An error occurred during prediction: {e}"}), 500

# ─── Flask app launcher with dynamic port ─────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use PORT env var if available
    app.run(debug=True, host='0.0.0.0', port=port)
