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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Data Loading & Preprocessing

def load_data(file_path: str) -> pd.DataFrame:
    """Loads commodity price data from a CSV file."""
    df = pd.read_csv(file_path, parse_dates=["Date"], dayfirst=True)
    df.set_index("Date", inplace=True)
    print(f"Data Loaded Successfully from {file_path}!")
    return df

def load_gdp_data(gdp_file: str) -> pd.DataFrame:
    """
    Processes annual GDP data to match daily oil price data.
    Assigns the same GDP value to all dates within a given year.
    
    :param gdp_file: Path to the GDP dataset
    :return: Processed GDP dataset with daily values for each year.
    """
    # Load the GDP data
    gdp_df = pd.read_csv(gdp_file)

    # Clean the data: Remove non-numeric characters ($, B, %, commas)
    gdp_df["GDP"] = gdp_df["GDP"].replace({r'\$': '', 'B': '', ',': ''}, regex=True).astype(float)
    gdp_df["GDP per Capita"] = gdp_df["GDP per Capita"].replace({r'\$': '', ',': ''}, regex=True).astype(float)
    gdp_df["Growth"] = gdp_df["Growth"].replace({'%': ''}, regex=True).astype(float)

    # Convert 'Year' column to datetime
    gdp_df["Year"] = pd.to_datetime(gdp_df["Year"], format='%Y')

    # Expand GDP values to all days within the year
    gdp_df = gdp_df.set_index("Year").resample("D").ffill()

    print("\nGDP Data Expanded to Daily Frequency:")
    print(gdp_df.head())

    return gdp_df


def merge_oil_gdp(oil_df: pd.DataFrame, gdp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges daily oil price data with annual GDP data.
    Ensures that each oil price entry gets the correct GDP value from its corresponding year.
    
    :param oil_df: Oil price dataset (daily frequency)
    :param gdp_df: GDP dataset with values expanded to daily frequency
    :return: Merged dataset
    """
    # Merge using the index (Date for oil, Year expanded to daily for GDP)
    merged_df = oil_df.merge(gdp_df, left_index=True, right_index=True, how="left")

    print("\nMerged dataset preview:")
    print(merged_df.head())

    return merged_df


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
    """Performs Augmented Dickey-Fuller Test after handling NaNs and Infs."""
    
    # Drop NaN and Infinite values
    series = df[column].replace([np.inf, -np.inf], np.nan).dropna()

    if series.empty:
        print(f"Column {column} is empty after removing NaNs/Infs. Cannot perform ADF test.")
        return False
    
    # Perform ADF test
    result = adfuller(series)
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

# function for backtesting LSTM model
def backtest_lstm(model, data, scaler, lookback=60):
    """Function to backtest LSTM predictions"""
    
    data_scaled = scaler.transform(data.values)
    
    predictions = []
    
    for i in range(lookback, len(data)):
        # Prepare the input sequence
        sequence = data_scaled[i-lookback:i, :]
        
        # Make prediction for the next time step
        predicted = model.predict(sequence.reshape(1, lookback, data.shape[1]), verbose=0)  
        predictions.append(predicted[0, 0])

    # Fix inverse transform issue
    predictions_rescaled = scaler.inverse_transform(
        np.concatenate([np.array(predictions).reshape(-1, 1), np.zeros((len(predictions), 1))], axis=1)
    )[:, 0]  # Extract only the first column (Price)

    # Plot results for comparison
    plt.plot(data.index[lookback:], data["Price"].iloc[lookback:], label="Actual")
    plt.plot(data.index[lookback:], predictions_rescaled, label="Predicted")
    plt.legend()
    plt.show()
    
    return predictions_rescaled


#evaluate LSTM model
def evaluate_lstm(y_actual, y_pred):
    """Evaluates the LSTM model using RMSE, MAE, and R²."""
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return {"RMSE": rmse, "MAE": mae, "R2": r2}



#  Save & Export Results
def save_results_to_csv(df: pd.DataFrame, filename="commodity_gdp_analysis_results.csv"):
    """Saves merged df to CSV."""
    df.to_csv(filename)
    print(f"Results saved to {filename}")
