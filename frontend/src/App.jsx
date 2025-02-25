import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend, CartesianGrid, ResponsiveContainer,
} from "recharts";

const App = () => {
  const [metrics, setMetrics] = useState({});
  const [forecast, setForecast] = useState(null);
  const [priceData, setPriceData] = useState([]);

  useEffect(() => {
    // Fetch Model Evaluation Metrics
    axios.get("http://127.0.0.1:5000/metrics")
      .then(response => setMetrics(response.data))
      .catch(error => console.error("Error fetching metrics:", error));

    // Fetch Forecasted Price
    axios.get("http://127.0.0.1:5000/forecast")
      .then(response => setForecast(response.data.forecasted_price))
      .catch(error => console.error("Error fetching forecast:", error));

    // Fetch Historical Prices (for plotting)
    axios.get("http://127.0.0.1:5000/historical_prices")
      .then(response => setPriceData(response.data))
      .catch(error => console.error("Error fetching historical prices:", error));
  }, []);

  return (
    <div className="dashboard">
      <h1>📈 Oil Price Forecasting Dashboard</h1>

      {/* Model Metrics */}
      <div className="metrics">
        <h2>📊 Model Performance</h2>
        <p><strong>RMSE:</strong> {metrics.RMSE}</p>
        <p><strong>MAE:</strong> {metrics.MAE}</p>
        <p><strong>R² Score:</strong> {metrics.R2}</p>
      </div>

      {/* Forecasted Price */}
      <div className="forecast">
        <h2>🔮 Forecasted Oil Price</h2>
        <p><strong>Next Predicted Price:</strong> {forecast ? `$${forecast.toFixed(2)}` : "Loading..."}</p>
      </div>

      {/* Historical Price Trends */}
      <div className="chart">
        <h2>📉 Historical Oil Price Trends</h2>
        <ResponsiveContainer width="90%" height={400}>
          <LineChart data={priceData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="actual_price" stroke="#8884d8" name="Actual Price" />
            <Line type="monotone" dataKey="predicted_price" stroke="#82ca9d" name="Predicted Price" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Price Change Visualization */}
      <div className="chart">
        <h2>📊 Daily Price Change</h2>
        <ResponsiveContainer width="90%" height={400}>
          <LineChart data={priceData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="price_change" stroke="#ff7300" name="Daily Change" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default App;
