import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def load_data(file_type='prediction'):
    if file_type == 'prediction':
        filepaths = ['data/1979-2021.csv']
    elif file_type == 'visualization':
        filepaths = [
            'data/cleaned_data_20_2025.csv',
            'data/cleaned_data_2023-2024.csv',
            'data/cleaned_data_2024-2025.csv'
        ]
    else:
        raise ValueError("Invalid file_type specified")

    dfs = []
    for fp in filepaths:
        if not os.path.exists(fp):
            print(f"Warning: File not found - {fp}")
            continue
        
        try:
            df = pd.read_csv(fp)
            required_cols = ['Date', 'USD_Price', 'Country']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing columns {missing} in file {fp}")
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {fp}: {str(e)}")
    
    if not dfs:
        raise FileNotFoundError("No valid data files were loaded.")
    
    data = pd.concat(dfs, ignore_index=True)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date').drop_duplicates('Date')
    data.set_index('Date', inplace=True)
    data = data.ffill().bfill()
    return data

def validate_model(data):
    if data.empty:
        raise ValueError("Gold price data is empty")
    
    X = data.drop(['USD_Price'], axis=1, errors='ignore')
    y = data['USD_Price']
    
    if X.empty:
        raise ValueError("No valid features in data")
    
    min_samples = 10
    n_splits = max(2, len(data) // min_samples)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        max_features=0.8,
        random_state=42
    )
    scaler = StandardScaler()
    
    scores = []
    mae_values = []
    mse_values = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        scores.append(r2_score(y_test, y_pred))
        mae_values.append(mean_absolute_error(y_test, y_pred))
        mse_values.append(mean_squared_error(y_test, y_pred))
    
    avg_r2 = np.mean(scores) if scores else None
    return {
        'metrics': {
            'avg_r2': avg_r2,
            'mae_values': mae_values,
            'mse_values': mse_values
        }
    }

def create_sequences(data, window_size=7):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, 0])
        y.append(data[i+window_size, 0])
    return np.expand_dims(np.array(X), -1), np.array(y)

def train_model(data, country):
    country_data = data[data['Country'] == country]
    scaled_prices = country_data[['USD_Price']].values
    X, y = create_sequences(scaled_prices)
    
    split_idx = int(len(X) * 0.9)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    model = create_lstm_model((X_train.shape[1], 1))
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(X_train, y_train, epochs=50, 
                        validation_data=(X_test, y_test),
                        callbacks=[early_stop],
                        batch_size=32)
    return model, history

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(50),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='mse', 
                  metrics=['mae'])
    return model

def predict_future_prices(model, country_data, future_days=30):
    last_data = country_data[['USD_Price']].values[-7:]
    predictions = []

    for _ in range(future_days):
        prediction = model.predict(last_data.reshape(1, -1, 1))
        predictions.append(prediction[0][0])
        last_data = np.append(last_data[1:], prediction)

    return predictions
