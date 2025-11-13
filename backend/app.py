import numpy as np
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)
CORS(app)

# Load models and scalers
lstm_model = load_model("models/lstm_model.h5")
lstm_scaler = joblib.load("models/lstm_scaler.save")

xgb_model = joblib.load("models/xgb_btc_model.pkl")
xgb_scaler = joblib.load("models/xgb_scaler.pkl")


# -------------------------------
# 🔵 LSTM Prediction API (old)
# -------------------------------
@app.route("/predict_lstm", methods=["POST"])
def predict_lstm():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": 365, "interval": "daily"}
    r = requests.get(url, params=params)

    prices = [p[1] for p in r.json()['prices']]
    if len(prices) < 60:
        return jsonify({"error": "Not enough data to predict"})

    prices_scaled = lstm_scaler.transform(np.array(prices).reshape(-1, 1)).flatten()
    X_input = np.array(prices_scaled[-60:]).reshape(1, 60, 1)
    pred_norm = lstm_model.predict(X_input)[0][0]
    pred_price = lstm_scaler.inverse_transform([[pred_norm]])[0][0]

    return jsonify({"predicted_price": float(pred_price)})


# -------------------------------
# 🟢 XGBoost Prediction API (old)
# -------------------------------
@app.route("/predict_xgb", methods=["POST"])
def predict_xgb():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": 365, "interval": "daily"}
    r = requests.get(url, params=params)

    prices = [p[1] for p in r.json()['prices']]
    if len(prices) < 7:
        return jsonify({"error": "Not enough data to predict"})

    lags = [prices[-i] for i in range(1, 8)]
    X_input = np.array(lags).reshape(1, 7)
    X_input_scaled = xgb_scaler.transform(X_input)
    pred = xgb_model.predict(X_input_scaled)[0]

    return jsonify({"predicted_price": float(pred)})


# -------------------------------------------------------
# ⭐ NEW API: Return History + Predicted → For Graph ⭐
# -------------------------------------------------------
@app.route("/predict_with_history", methods=["POST"])
def predict_with_history():
    req = request.json
    model_type = req.get("model")

    # Fetch 365 days data
    r = requests.get(
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
        params={"vs_currency": "usd", "days": 365, "interval": "daily"}
    ).json()

    prices = [p[1] for p in r["prices"]]

    # -----------------------
    # LSTM prediction
    # -----------------------
    if model_type == "lstm":
        scaled = lstm_scaler.transform(np.array(prices).reshape(-1, 1)).flatten()
        X_input = np.array(scaled[-60:]).reshape(1, 60, 1)
        pred_norm = lstm_model.predict(X_input)[0][0]
        pred_price = lstm_scaler.inverse_transform([[pred_norm]])[0][0]

    # -----------------------
    # XGBoost prediction
    # -----------------------
    elif model_type == "xgboost":
        lags = [prices[-i] for i in range(1, 8)]
        X = np.array(lags).reshape(1, 7)
        X_scaled = xgb_scaler.transform(X)
        pred_price = xgb_model.predict(X_scaled)[0]

    else:
        return jsonify({"error": "Invalid model"}), 400

    return jsonify({
        "history": prices,
        "predicted_price": float(pred_price)
    })


if __name__ == "__main__":
    app.run(port=5000, debug=True)
