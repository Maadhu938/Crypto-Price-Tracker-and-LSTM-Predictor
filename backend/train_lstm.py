import numpy as np
import requests
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import joblib

print("---- Training LSTM model ----")

# Create directory if not exists
os.makedirs("models", exist_ok=True)

# 1. Get Bitcoin price data
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
params = {"vs_currency": "usd", "days": 365, "interval": "daily"}
r = requests.get(url, params=params)
prices = [p[1] for p in r.json()['prices']]

# 2. Normalize prices
scaler = MinMaxScaler()
prices_norm = scaler.fit_transform(np.array(prices).reshape(-1, 1))

# 3. Prepare sequences
LOOKBACK = 60
X, y = [], []
for i in range(LOOKBACK, len(prices_norm)):
    X.append(prices_norm[i - LOOKBACK:i, 0])
    y.append(prices_norm[i, 0])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# 4. Train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(LOOKBACK, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 6. Train model
print("Training LSTM... epochs=50, batch=64")
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# 7. Evaluate model
y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
rmse = sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
mape = np.mean(np.abs((y_test_rescaled - y_pred_rescaled) / y_test_rescaled)) * 100
r2 = r2_score(y_test_rescaled, y_pred_rescaled)

print("\nEvaluation Metrics:")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²:   {r2:.4f}")

# 8. Save model and scaler
model.save("models/lstm_model.h5")
joblib.dump(scaler, "models/lstm_scaler.save")
print("Model and scaler saved to 'models/' directory.")
print("LSTM training complete.")
