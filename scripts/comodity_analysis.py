#  Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


# Data Loading & Preprocessing

def load_data(file_path: str) -> pd.DataFrame:
    """Loads commodity price data from a CSV file."""
    df = pd.read_csv(file_path, parse_dates=["Date"], dayfirst=True)
    df.set_index("Date", inplace=True)
    print(f"Data Loaded Successfully from {file_path}!")
    return df

def load_gdp_data(gdp_file: str, method="interpolate") -> pd.DataFrame:
    """
    Processes annual GDP data to match daily oil price data.
    :param gdp_file: Path to the GDP dataset
    :param method: "interpolate" (convert to monthly/daily) or "repeat" (assign annual value to each day)
    :return: Processed GDP dataset
    """
    gdp_df = pd.read_csv(gdp_file)

    # Convert 'Year' column to datetime (set to January 1st of each year)
    gdp_df["Year"] = pd.to_datetime(gdp_df["Year"].astype(str) + "-01-01")
    gdp_df.set_index("Year", inplace=True)

    if method == "interpolate":
        gdp_df = gdp_df.resample('M').interpolate()  # Convert annual to monthly
        print("GDP Data Converted to Monthly Frequency via Interpolation.")
    elif method == "repeat":
        print("GDP Data Assigned to Each Day in Corresponding Year.")
    else:
        raise ValueError("Method must be 'interpolate' or 'repeat'.")
    
    return gdp_df


# Exploratory Data Analysis (EDA)

def visualize_data(df: pd.DataFrame, title="Oil Prices and GDP"):
    """Plots time series data."""
    plt.figure(figsize=(12, 5))
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

def correlation_analysis(df: pd.DataFrame):
    """Displays correlation matrix between oil prices and GDP."""
    plt.figure(figsize=(8, 5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

def check_stationarity(df: pd.DataFrame, column: str) -> bool:
    """Performs Augmented Dickey-Fuller Test."""
    result = adfuller(df[column])
    print(f"ADF Statistic for {column}: {result[0]}")
    print(f"p-value: {result[1]}")
    return result[1] <= 0.05  # Returns True if stationary


#  Model Implementation

def fit_var_model(df: pd.DataFrame, lags=5):
    """Fits a VAR model to analyze GDP and oil price relationships."""
    model = VAR(df)
    result = model.fit(lags)
    print(result.summary())
    return result

def fit_markov_switching(df: pd.DataFrame):
    """Fits a Markov-Switching ARIMA model for oil market conditions."""
    model = MarkovRegression(df, k_regimes=2, trend="c", switching_variance=True)
    result = model.fit()
    print(result.summary())
    return result


# LSTM Model

def scale_data(df: pd.DataFrame):
    """Scales data for LSTM processing."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(df.values), scaler

def prepare_lstm_data(data, lookback=60):
    """Prepares sequences for LSTM training."""
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """Builds an LSTM model for time series forecasting."""
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def train_lstm(df: pd.DataFrame, lookback=60, epochs=20, batch_size=16):
    """Trains an LSTM model on the given data."""
    data_scaled, scaler = scale_data(df)
    X_train, y_train = prepare_lstm_data(data_scaled, lookback)
    model = build_lstm_model((lookback, df.shape[1]))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model, scaler


#  Save & Export Results

def save_results_to_csv(df: pd.DataFrame, filename="commodity_gdp_analysis_results.csv"):
    """Saves model outputs to CSV."""
    df.to_csv(filename)
    print(f"Results saved to {filename}")
