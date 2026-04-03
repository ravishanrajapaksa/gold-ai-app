import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta

# ✅ PAGE TITLE
st.title("GLD Trading Dashboard")

# ✅ LOAD DATA
st.write("Loading data...")
data = yf.download("GLD", period="90d", interval="1h")

# ✅ FIX MULTIINDEX (important for yfinance)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# ✅ CHECK IF DATA LOADED
if data.empty:
    st.error("❌ Failed to load data from yfinance")
else:
    close = data['Close']

    # ✅ ADD INDICATORS
    data['SMA'] = ta.trend.sma_indicator(close, window=14)
    data['EMA'] = ta.trend.ema_indicator(close, window=14)
    data['RSI'] = ta.momentum.rsi(close, window=14)

    data = data.dropna()

    # ✅ TARGET
    data['Target'] = close.shift(-3)
    data = data.dropna()

    # ✅ DISPLAY DATA
    st.subheader("Latest Data")
    st.write(data.tail())

    # ✅ CHART
    st.subheader("Price Chart")
    st.line_chart(data[['Close', 'SMA', 'EMA']])

    # ✅ RSI CHART
    st.subheader("RSI")
    st.line_chart(data['RSI'])
