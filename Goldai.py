import streamlit as st
import pandas as pd
import numpy as np
from tvDatafeed import TvDatafeed, Interval
import ta
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🤖 AI Trading Dashboard - XGBoost + LSTM Ensemble")

tv = TvDatafeed()

@st.cache_data
def load_data(symbol='CL1', exchange='NYMEX', bars=1000):
    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_1_hour, n_bars=bars)
    if df is None or df.empty:
        return None
    df = df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
    # Indicators
    df['SMA'] = ta.trend.sma_indicator(df['Close'], window=14)
    df['EMA'] = ta.trend.ema_indicator(df['Close'], window=14)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd(df['Close'])
    df['BB_High'] = ta.volatility.bollinger_hband(df['Close'])
    df['BB_Low'] = ta.volatility.bollinger_lband(df['Close'])
    df = df.dropna()
    return df

data = load_data()
if data is None:
    st.error("❌ Failed to load data")
else:
    st.subheader("Latest Data")
    st.dataframe(data.tail(10))

    # ---------------- LSTM for Price Prediction ----------------
    lstm_features = ['Close','SMA','EMA','RSI','MACD','BB_High','BB_Low']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[lstm_features])

    lookback = 10
    X_lstm, y_lstm = [], []
    for i in range(lookback, len(scaled_data)-3):
        X_lstm.append(scaled_data[i-lookback:i])
        y_lstm.append(scaled_data[i+3,0])
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

    split = int(0.8*len(X_lstm))
    X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
    y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]

    lstm_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=32, validation_data=(X_test_lstm, y_test_lstm), verbose=0)

    pred_scaled = lstm_model.predict(X_lstm)
    pred_prices = scaler.inverse_transform(np.concatenate([pred_scaled, np.zeros((pred_scaled.shape[0], scaled_data.shape[1]-1))], axis=1))[:,0]
    data = data.iloc[lookback+3:].copy()
    data['LSTM_Pred'] = pred_prices

    # ---------------- XGBoost for Buy/Sell ----------------
    features = ['Close','SMA','EMA','RSI','MACD','BB_High','BB_Low']
    data['Target'] = np.where(data['Close'].shift(-3) < data['Close'], 0, 1)
    X = data[features]
    y = data['Target']

    xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.1)
    xgb_model.fit(X, y)
    data['XGB_Signal'] = xgb_model.predict(X)

    # ---------------- Ensemble Signal ----------------
    def ensemble(row):
        if row['XGB_Signal']==1 and row['LSTM_Pred'] > row['Close']:
            return 1
        elif row['XGB_Signal']==0 and row['LSTM_Pred'] < row['Close']:
            return 0
        else:
            return -1
    data['Ensemble_Signal'] = data.apply(ensemble, axis=1)

    # ---------------- Candlestick + Signals ----------------
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price'))
    fig.add_trace(go.Scatter(x=data.index, y=data['LSTM_Pred'], line=dict(color='orange'), name='LSTM Prediction'))

    buy = data[data['Ensemble_Signal']==1]
    sell = data[data['Ensemble_Signal']==0]

    fig.add_trace(go.Scatter(x=buy.index, y=buy['Close'], mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name='Buy'))
    fig.add_trace(go.Scatter(x=sell.index, y=sell['Close'], mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name='Sell'))

    st.plotly_chart(fig, use_container_width=True)

    # ---------------- Next Signal ----------------
    latest = data.iloc[-1]
    if latest['Ensemble_Signal']==1:
        st.success(f"Next Signal: BUY | Predicted Price: {latest['LSTM_Pred']:.2f}")
    elif latest['Ensemble_Signal']==0:
        st.error(f"Next Signal: SELL | Predicted Price: {latest['LSTM_Pred']:.2f}")
    else:
        st.info(f"Next Signal: HOLD | Predicted Price: {latest['LSTM_Pred']:.2f}")

    # ---------------- Backtesting ----------------
    data['Strategy_Return'] = np.where(data['Ensemble_Signal']==1, data['Close'].pct_change().shift(-1), 0)
    data['Cumulative_Return'] = (1 + data['Strategy_Return']).cumprod()

    st.subheader("Equity Curve")
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=data.index, y=data['Cumulative_Return'], name='Equity Curve'))
    st.plotly_chart(fig_eq, use_container_width=True)

    total_return = data['Cumulative_Return'].iloc[-1]-1
    win_rate = (data['Strategy_Return']>0).sum() / (data['Strategy_Return']!=0).sum()
    st.write(f"Total Return: {total_return:.2%}")
    st.write(f"Win Rate: {win_rate:.2%}")
