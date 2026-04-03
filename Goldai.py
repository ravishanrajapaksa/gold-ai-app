import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import ta

# ✅ STEP 1: LOAD DATA
data = yf.download("GLD", period="90d", interval="1h")

# ✅ FIX: Flatten MultiIndex columns (important!)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# ✅ STEP 2: USE CLEAN 1D SERIES
close = data['Close']

# ✅ ADD INDICATORS
data['SMA'] = ta.trend.sma_indicator(close, window=14)
data['EMA'] = ta.trend.ema_indicator(close, window=14)
data['RSI'] = ta.momentum.rsi(close, window=14)

data = data.dropna()

# ✅ STEP 3: CREATE TARGET
data['Target'] = close.shift(-3)

data = data.dropna()
