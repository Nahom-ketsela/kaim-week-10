# Brent Oil Price Analysis & Event Impact Dashboard

## Overview

This project aims to analyze how significant political and economic events influence the price of Brent oil over time. Using historical Brent oil price data (from 1987 to 2022), the analysis leverages time series models, change point detection, and machine learning techniques to identify key events that affect oil prices. The final insights are made accessible through an interactive dashboard built with Flask (backend) and React (frontend), providing a user-friendly interface for exploring the data and correlations.

### Business Objective
The goal is to help investors, policymakers, and energy companies understand how external factors (e.g., political decisions, economic sanctions, OPEC policies) impact oil prices. The project provides data-driven insights that can guide investment strategies, policy development, and operational planning in the energy sector.

---

## Key Features

- **Time Series Analysis**: Utilizes ARIMA and other econometric models to analyze historical price data and identify significant trends.
- **Change Point Detection**: Identifies key events that caused shifts in oil prices.
- **Interactive Dashboard**: Built with Flask and React to visualize price trends and event correlations.
- **Event Correlation**: Allows users to explore the relationship between major global events and price fluctuations.
- **Forecasting**: Provides predictive insights using advanced statistical models and machine learning techniques.

---

## Project Workflow

### 1. **Data Analysis & Preprocessing**
   - Historical Brent oil price data is cleaned, preprocessed, and transformed into a format suitable for time series analysis.
   - Key economic and political events are identified and linked to the data to analyze their impact on price changes.
   - Time series models such as ARIMA and advanced methods (e.g., VAR, Markov-Switching ARIMA) are applied to study the relationship between oil prices and various influencing factors.

### 2. **Model Building**
   - Statistical and econometric models are used to detect changes and forecast future trends.
   - Machine learning algorithms like LSTM (Long Short-Term Memory) networks are explored to capture complex dependencies.
   - The models evaluate the impact of external factors like GDP, inflation, exchange rates, and political events on Brent oil prices.

### 3. **Dashboard Development**
   - A Flask backend is developed to serve the analysis results via APIs.
   - A React frontend visualizes the results with interactive charts, event highlights, and filters to explore how specific events have impacted oil prices.
   - Features include date-range selections, event-specific visualizations, and model performance metrics.

---

## How to Use

### 1. **Clone the Repository**
```bash
git clone <this-repo>
cd brent-oil-analysis-dashboard
```

### 2. **Set Up the Environment**
   - Install dependencies for the backend (Flask) and frontend (React).

#### Backend (Flask)
```bash
cd backend
pip install -r requirements.txt
```

#### Frontend (React)
```bash
cd frontend
npm install
```

### 3. **Run the Application**

#### Start the Flask Backend
```bash
cd backend
python app.py
```

#### Start the React Frontend
```bash
cd frontend
npm start
```

The application will be available at `http://localhost:3000` for the frontend and `http://localhost:5000` for the backend.

---

## Structure

- **/backend**: Contains the Flask API that serves the analysis results and connects to the data processing pipeline.
- **/frontend**: Contains the React-based dashboard for visualizing and interacting with the results.
- **/data**: Raw and processed datasets for analysis.
- **/models**: Python scripts for statistical modeling and time series analysis.
- **/notebooks**: Jupyter notebooks for exploratory data analysis and model development.

---

## Insights & Use Cases

- **Event Impact Analysis**: Use the dashboard to visualize how major global events (e.g., OPEC decisions, geopolitical conflicts) affect Brent oil prices.
- **Trend Forecasting**: Leverage time series models to predict future price movements based on historical data.
- **Interactive Exploration**: Drill down into specific events and time periods to understand their impact in detail.

---


