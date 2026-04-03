import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(layout="wide")
st.title("AI Gold Trading System")

# Auto refresh every 60 seconds
st.caption("Auto-refreshes every 60 seconds")

# Fetch data
data = yf.download("GLD", period="120d", interval="1h")

# Indicators
data['SMA'] = ta.trend.sma_indicator(data['Close'], window=14)
data['EMA'] = ta.trend.ema_indicator(data['Close'], window=14)
data['RSI'] = ta.momentum.rsi(data['Close'], window=14)

data = data.dropna()

# Scale data
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data[['Close']])

# Prepare sequences
X = []
y = []

window = 24  # last 24 hours

for i in range(window, len(scaled)):
    X.append(scaled[i-window:i])
    y.append(scaled[i])

X, y = np.array(X), np.array(y)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train (quick training)
model.fit(X, y, epochs=2, batch_size=32, verbose=0)

# Predict next value
last_sequence = scaled[-window:]
last_sequence = np.reshape(last_sequence, (1, window, 1))

predicted = model.predict(last_sequence)
predicted_price = scaler.inverse_transform(predicted)[0][0]

current_price = data['Close'].iloc[-1]

# Generate signal
if predicted_price > current_price:
    signal = "BUY"
elif predicted_price < current_price:
    signal = "SELL"
else:
    signal = "HOLD"

# UI
col1, col2, col3 = st.columns(3)

col1.metric("Current Price", round(current_price, 2))
col2.metric("Predicted Price", round(predicted_price, 2))
col3.metric("Signal", signal)

# Chart
st.line_chart(data['Close'])

# Extra info
st.subheader("Indicators")
st.write(data[['Close', 'SMA', 'EMA', 'RSI']].tail())
