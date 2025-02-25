from flask import Flask, jsonify, render_template, request
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and scaler
lstm_model = load_model("../notebooks/lstm_price_forecasting.keras")  # Use Keras' load_model for .h5 format
scaler = joblib.load("../notebooks/lstm_scaler.pkl")

# Load preprocessed data
data = pd.read_csv("../notebooks/commodity_gdp_analysis_results.csv")
data.columns = [col.strip() for col in data.columns]  # Clean column names

# Main page route
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/metrics", methods=["GET"])
def get_model_metrics():
    """Returns evaluation metrics for the LSTM model."""
    
    # Generate predictions using the trained model
    y_actual = data["Price"].values
    y_pred = generate_predictions(data[["Price", "GDP"]])

    metrics = evaluate_lstm(y_actual, y_pred)
    return jsonify(metrics)


@app.route("/forecast", methods=["GET"])
def get_forecast():
    """Returns the LSTM model forecast for the next period."""
    last_n_days = data.iloc[-60:][["Price", "GDP"]].values
    last_n_days_scaled = scaler.transform(last_n_days)
    last_n_days_scaled = np.expand_dims(last_n_days_scaled, axis=0)

    # Generate prediction for the next period
    prediction = lstm_model.predict(last_n_days_scaled)
    
    # Manually add the GDP value (take the last GDP value from the data)
    last_gdp = last_n_days[-1, 1]  # Last GDP value in the dataset
    prediction_with_gdp = np.array([[prediction[0][0], last_gdp]])  # Combine the prediction with the GDP value

    # Inverse transform using the scaler (since the scaler was trained on both Price and GDP)
    prediction_final = scaler.inverse_transform(prediction_with_gdp)[0][0]

    return jsonify({"forecasted_price": prediction_final})


@app.route("/historical_prices", methods=["GET"])
def get_historical_prices():
    """Returns historical actual vs. predicted prices for visualization."""
    df = pd.read_csv("../notebooks/commodity_gdp_analysis_results.csv")
    df.columns = [col.strip() for col in df.columns]  # Clean column names

    # Generate predictions and add them to the dataframe
    df["Predicted_Price"] = generate_predictions(df[["Price", "GDP"]])

    # Calculate daily price change
    df["price_change"] = df["Price"].diff().fillna(0)

    return jsonify(df.to_dict(orient="records"))


def generate_predictions(data_subset):
    """Generates predictions for each row in the dataset."""
    data_scaled = scaler.transform(data_subset)  # Scale all data

    predictions = []
    for i in range(len(data_scaled)):  # Iterate over each row
        input_data = np.expand_dims(data_scaled[i], axis=(0, 1))  # Reshape for LSTM: (1, 1, features)
        pred = lstm_model.predict(input_data)[0, 0]  # Predict and extract value
        predictions.append(pred)

    # Convert list to numpy array
    predictions = np.array(predictions).reshape(-1, 1)  # Reshape to match scaler expectations

    # Create a placeholder array with the same shape as input (Price, GDP)
    predictions_padded = np.zeros((len(predictions), 2))  
    predictions_padded[:, 0] = predictions[:, 0]  # Assign predicted prices

    # Inverse transform the predictions (restore original scale)
    predictions_inverse = scaler.inverse_transform(predictions_padded)[:, 0]  # Extract only price column

    return predictions_inverse  # Return full array of predicted prices


def evaluate_lstm(y_actual, y_pred):
    """Evaluates LSTM model using RMSE, MAE, and R-squared."""
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }


if __name__ == "__main__":
    app.run(debug=True)
