import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import ruptures as rpt


#  Data Loading & Cleaning

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads Brent oil price data from CSV and returns a DataFrame.
    
    :param file_path: The path to the CSV file.
    :return: A pandas DataFrame with columns ["Date", "Price"].
    """
    df = pd.read_csv(file_path, parse_dates=["Date"], dayfirst=True)
    print("Loaded Sucessfully.✅ \n") 
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset by handling missing values, sorting by date, etc.
    
    :param df: Raw DataFrame with "Date" and "Price" columns.
    :return: Cleaned DataFrame.
    """
    df.dropna(subset=["Price"], inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%y", errors="coerce")
    df.sort_values(by="Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    
    print("Cleaned DataFrame\n") 
    print("Cleaned Data:\n", df.head())  
    
    return df


# Exploratory Data Analysis

def plot_price_series(df: pd.DataFrame) -> None:
    """
    Plots the Brent Oil Price time series.
    
    :param df: Cleaned DataFrame.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["Price"], label="Brent Oil Price", color='blue')
    plt.title("Brent Oil Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price (USD/barrel)")
    plt.legend()
    plt.show()
    

def check_stationarity(df: pd.DataFrame) -> None:
    """
    Conducts the Augmented Dickey-Fuller (ADF) test to check for stationarity.
    
    :param df: DataFrame with "Price" column.
    """
    adf_result = adfuller(df["Price"].dropna())
    print("\nAugmented Dickey-Fuller Test:")
    print(f"ADF Statistic: {adf_result[0]}")
    print(f"p-value: {adf_result[1]}")
    for key, value in adf_result[4].items():
        print(f"Critical Values {key}: {value}")


# Time Series Modeling (ARIMA)
def fit_arima_model(df: pd.DataFrame, order=(1,1,1)):
    """
    Fits an ARIMA model with the specified order.
    
    :param df: DataFrame with "Date" and "Price" columns.
    :param order: (p, d, q) tuple for ARIMA model parameters.
    :return: Fitted ARIMA model results.
    """
    # Ensure "Date" is in datetime format and set it as the index
    df["Date"] = pd.to_datetime(df["Date"])
    df_ts = df.set_index("Date")

    # Set frequency (assuming daily data)
    df_ts = df_ts.asfreq('D')  

    # Fit ARIMA model
    model = sm.tsa.ARIMA(df_ts["Price"], order=order)
    results = model.fit()
    
    return results


# Change Point Detection

def detect_change_points(df: pd.DataFrame, penalty: float = 10.0):
    """
    Detects change points in the 'Price' series using the PELT algorithm.
    
    :param df: DataFrame with "Price" column.
    :param penalty: Penalty term for controlling number of breakpoints.
    :return: List of indices representing the breakpoints.
    """
    price_array = df["Price"].values
    model = rpt.Pelt(model="rbf").fit(price_array)
    breakpoints = model.predict(pen=penalty)
    return breakpoints

def plot_change_points(df: pd.DataFrame, breakpoints: list) -> None:
    """
    Visualizes the Brent oil price series with detected change points.
    
    :param df: DataFrame with "Price" column.
    :param breakpoints: List of indices representing the breakpoints.
    """
    price_array = df["Price"].values
    rpt.display(price_array, breakpoints)
    plt.title("Detected Change Points in Brent Oil Price")
    plt.show()
