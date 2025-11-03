import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt  # <-- Added for RMSE calculation
import joblib

print("----Training XGBoost model----")

# Data collection: last 365 days' daily BTC close prices from CoinGecko
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
params = {"vs_currency": "usd", "days": 365, "interval": "daily"}
r = requests.get(url, params=params)
prices = [price[1] for price in r.json()['prices']]

# Create dataframe and lag features
df = pd.DataFrame({"Close": prices})
for i in range(1, 8):  # lag1 to lag7 as features
    df[f"lag_{i}"] = df["Close"].shift(i)

df = df.dropna().reset_index(drop=True)

X = df[[f"lag_{i}" for i in range(1, 8)]]
y = df["Close"]

# MinMax scale features as per paper
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split 80:20 (no shuffle because time series!)
split = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y[:split], y[split:]

# XGBoost model with paper-stated hyperparameters
model = XGBRegressor(
    learning_rate=0.1,
    n_estimators=150,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))  # <-- Correct RMSE calculation
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
r2 = r2_score(y_test, y_pred)

print("Evaluation Metrics (formulas per paper):")
print("MAE  = (1/n) * sum |y_i - ŷ_i|")
print("RMSE = sqrt((1/n)*sum(y_i - ŷ_i)^2)")
print("MAPE = (1/n) * sum( |(y_i - ŷ_i) / y_i| ) * 100")
print("R²   = 1 - sum((y_i - ŷ_i)^2) / sum((y_i - mean_y)^2)")

print(f"\nMAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²:   {r2:.4f}")

# Save model + scaler
joblib.dump(model, "models/xgb_btc_model.pkl")
joblib.dump(scaler, "models/xgb_scaler.pkl")
print("Model and scaler saved in 'models/' folder.")
print("XGBoost training complete.")