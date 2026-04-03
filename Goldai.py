import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# =========================
# Streamlit Page
# =========================
st.set_page_config(page_title="Gold AI Dashboard", layout="wide")
st.title("💰 Gold AI Trading Dashboard")

# =========================
# Load Data
# =========================
st.write("Fetching Gold data (GC=F, 1h)...")
data = yf.download("GC=F", period="90d", interval="1h")

if data.empty:
    st.error("❌ Failed to load data.")
else:
    close = data['Close']

    # =========================
    # Technical Indicators
    # =========================
    data['SMA'] = ta.trend.sma_indicator(close, window=14)
    data['EMA'] = ta.trend.ema_indicator(close, window=14)
    data['RSI'] = ta.momentum.rsi(close, window=14)
    data = data.dropna()

    # =========================
    # Target for ML (next 3 periods)
    # =========================
    data['Target'] = np.where(close.shift(-3) > close, 1, 0)
    data = data.dropna()

    st.subheader("Latest Data")
    st.dataframe(data.tail(10))

    # =========================
    # Charts
    # =========================
    st.subheader("Price Chart with SMA & EMA")
    st.line_chart(data[['Close', 'SMA', 'EMA']])

    st.subheader("RSI")
    st.line_chart(data['RSI'])

    # =========================
    # Machine Learning Model
    # =========================
    st.subheader("AI Signals (XGBoost + LSTM Ensemble)")

    features = ['SMA', 'EMA', 'RSI']
    X = data[features].values
    y = data['Target'].values

    # XGBoost Classifier
    xgb_model = XGBClassifier(n_estimators=50, max_depth=3, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X, y)
    data['XGB_Pred'] = xgb_model.predict(X)

    # LSTM Model
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    lstm_model = Sequential()
    lstm_model.add(LSTM(50, input_shape=(X_lstm.shape[1], X_lstm.shape[2])))
    lstm_model.add(Dense(1, activation='sigmoid'))
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam')
    lstm_model.fit(X_lstm, y, epochs=5, batch_size=32, verbose=0)
    data['LSTM_Pred'] = lstm_model.predict(X_lstm).flatten()
    data['LSTM_Pred_Label'] = np.where(data['LSTM_Pred']>0.5,1,0)

    # Ensemble Signal: Buy if both XGB & LSTM predict 1
    data['Signal'] = np.where((data['XGB_Pred']==1) & (data['LSTM_Pred_Label']==1), 'Buy', 'Hold')
    data['Signal'] = np.where((data['XGB_Pred']==0) & (data['LSTM_Pred_Label']==0), 'Sell', data['Signal'])

    st.dataframe(data[['Close', 'XGB_Pred', 'LSTM_Pred_Label', 'Signal']].tail(10))
