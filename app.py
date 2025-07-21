import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import warnings
warnings.filterwarnings("ignore")

import sklearn.compose._column_transformer as _ct
class _RemainderColsList: pass
_ct._RemainderColsList = _RemainderColsList

app = Flask(__name__, template_folder='templates', static_folder='static')

BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, 'best_model.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    FEATURE_NAMES = list(model.feature_names_in_)
    print(f"Loaded model; expecting columns: {FEATURE_NAMES}")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}:", e)
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
        # Get form data
        data = {col: request.form[col] for col in FEATURE_NAMES}

        # Convert numeric fields
        for num_col in ['bedroom_number', 'bathroom_number', 'living_space',
                        'land_space', 'price_per_unit', 'postcode']:
            if num_col in data:
                data[num_col] = float(data[num_col])

        # Create single-row DataFrame
        df = pd.DataFrame([data], columns=FEATURE_NAMES)

        # Predict (returns scaled prediction, e.g. 15.19)
        scaled_price = model.predict(df)[0]

        # Rescale prediction (if you originally divided y by 10,000)
        raw_price = scaled_price * 10000

        return jsonify({
            'price': round(raw_price, 2),
            'formatted_price': "${:,.2f}".format(round(raw_price, 2))
        })

    except KeyError as e:
        return jsonify({'error': f"Missing input field: {e}"}), 400
    except Exception as e:
        return jsonify({'error': f"An error occurred during prediction: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
