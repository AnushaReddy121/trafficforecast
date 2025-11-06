import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Load dataset
data = pd.read_csv("data/traffic.csv", parse_dates=["timestamp"])
data.set_index("timestamp", inplace=True)

# Plot traffic flow
plt.figure(figsize=(10,5))
plt.plot(data["flow"], label="Traffic Flow")
plt.title("Traffic Flow Over Time")
plt.xlabel("Time")
plt.ylabel("Flow")
plt.legend()
plt.show()

# Split into train and test
train_size = int(len(data) * 0.8)
train, test = data["flow"][:train_size], data["flow"][train_size:]

# Fit ARIMA model
model = ARIMA(train, order=(2, 1, 2))
model_fit = model.fit()

# Make predictions
predictions = model_fit.forecast(steps=len(test))
error = np.sqrt(mean_squared_error(test, predictions))

print(f"RMSE: {error:.2f}")

# Plot results
plt.figure(figsize=(10,5))
plt.plot(test.index, test, label="Actual")
plt.plot(test.index, predictions, label="Predicted", color="red")
plt.title("Traffic Flow Forecast")
plt.legend()
plt.show()
