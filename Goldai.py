import streamlit as st
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# =========================
# Load your data
data = pd.read_csv('gold_prices.csv')  # or yfinance download

# Ensure 'Close' column exists
if 'Close' not in data.columns:
    st.error("Dataset must contain a 'Close' column.")
else:
    close = data['Close']

    # =========================
    # Technical indicators
    data['SMA'] = ta.trend.sma_indicator(close=close, window=14, fillna=True)
    data['EMA'] = ta.trend.ema_indicator(close=close, window=14, fillna=True)
    data['RSI'] = ta.momentum.rsi(close=close, window=14, fillna=True)

    # =========================
    # Prepare data for XGBoost
    features = data[['SMA', 'EMA', 'RSI']].fillna(0)
    target = (close.shift(-1) > close).astype(int).fillna(0)  # next day up/down

    model = XGBClassifier()
    model.fit(features, target)

    st.write("Data prepared and XGBoost model trained.")
    st.write(data.head())
