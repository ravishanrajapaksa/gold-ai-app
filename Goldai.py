from sklearn.ensemble import RandomForestRegressor

# Prepare features
data['Target'] = data['Close'].shift(-3)
data = data.dropna()

features = ['Close', 'SMA', 'EMA', 'RSI']
X = data[features]
y = data['Target']

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# Predict
latest_data = data[features].tail(1)
predicted_price = model.predict(latest_data)[0]
