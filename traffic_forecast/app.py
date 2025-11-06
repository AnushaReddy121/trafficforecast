import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import math

st.set_page_config(page_title="Traffic Flow Forecast", layout="wide")

# Title
st.title("🚦 Traffic Flow Forecasting using Time Series Analysis")

# Load dataset
data = pd.read_csv("data/traffic.csv")
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)

st.subheader("📊 Traffic Data Preview")
st.write(data.head())

# Plot traffic flow
st.subheader("📈 Traffic Flow Over Time")
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(data.index, data['flow'], label='Traffic Flow', color='blue')
ax.set_xlabel("Time")
ax.set_ylabel("Flow")
ax.legend()
st.pyplot(fig)

# Forecasting
train_size = int(len(data) * 0.8)
train, test = data['flow'][:train_size], data['flow'][train_size:]

# Fit ARIMA model
model = ARIMA(train, order=(2,1,2))
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test))

# Calculate RMSE
rmse = math.sqrt(mean_squared_error(test, forecast))
st.metric("Model Accuracy (RMSE)", f"{rmse:.2f}")

# Plot forecast
st.subheader("🔮 Forecast vs Actual")
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(train.index, train, label='Training Data')
ax2.plot(test.index, test, label='Actual Flow')
ax2.plot(test.index, forecast, label='Predicted Flow', linestyle='--', color='red')
ax2.set_xlabel("Time")
ax2.set_ylabel("Flow")
ax2.legend()
st.pyplot(fig2)

st.success("✅ Forecast generated successfully!")
