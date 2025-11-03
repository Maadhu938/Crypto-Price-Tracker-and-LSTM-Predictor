import numpy as np
import requests
from flask import Flask, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)
CORS(app)

# Load models and scalers
lstm_model = load_model("models/lstm_model.h5")
lstm_scaler = joblib.load("models/lstm_scaler.save")  # scaler to normalize prices for LSTM

xgb_model = joblib.load("models/xgb_btc_model.pkl")
xgb_scaler = joblib.load("models/xgb_scaler.pkl")   # scaler to scale features for XGBoost


@app.route("/predict_lstm", methods=["POST"])
def predict_lstm():
    # Fetch last 365 days close price data for consistency
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": 365, "interval": "daily"}
    r = requests.get(url, params=params)
    
    prices = [p[1] for p in r.json()['prices']]
    if len(prices) < 60:
        return jsonify({"error": "Not enough data to predict"})
    
    # Normalize prices using LSTM scaler
    prices_scaled = lstm_scaler.transform(np.array(prices).reshape(-1, 1)).flatten()
    
    # Take last 60 normalized prices for LSTM input
    X_input = np.array(prices_scaled[-60:]).reshape(1, 60, 1)
    pred_norm = lstm_model.predict(X_input)[0][0]
    # Inverse transform prediction to original price
    pred_price = lstm_scaler.inverse_transform([[pred_norm]])[0][0]
    return jsonify({"predicted_price": float(pred_price)})


@app.route("/predict_xgb", methods=["POST"])
def predict_xgb():
    # Fetch last 365 days close prices
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": 365, "interval": "daily"}
    r = requests.get(url, params=params)
    
    prices = [p[1] for p in r.json()['prices']]
    if len(prices) < 7:
        return jsonify({"error": "Not enough data to predict"})
    
    # Prepare lag features from last 7 prices (latest is lag 1, oldest lag 7)
    lags = [prices[-i] for i in range(1, 8)]
    X_input = np.array(lags).reshape(1, 7)
    X_input_scaled = xgb_scaler.transform(X_input)
    pred = xgb_model.predict(X_input_scaled)[0]
    return jsonify({"predicted_price": float(pred)})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
