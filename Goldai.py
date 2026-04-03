import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import ta

# ✅ STEP 1: LOAD DATA FIRST
data = yf.download("GLD", period="90d", interval="1h")

# ✅ STEP 2: ADD INDICATORS
data['SMA'] = ta.trend.sma_indicator(data['Close'], window=14)
data['EMA'] = ta.trend.ema_indicator(data['Close'], window=14)
data['RSI'] = ta.momentum.rsi(data['Close'], window=14)

data = data.dropna()

# ✅ STEP 3: THEN CREATE TARGET
data['Target'] = data['Close'].shift(-3)

data = data.dropna()
