import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. Configuration
# -------------------------------
CONFIG = {
    "data_years": [2022, 2023],
    "look_back": 6,
    "train_ratio": 0.8,
    "seasonal_period": 24, # Daily seasonality
    "rolling_window": 24   # Update model every 24 hours
}

# -------------------------------
# 2. Data Engineering Module
# -------------------------------
class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.exog_cols = ["ghi", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]

    def load_and_process(self, data_dir):
        # Paths
        load_path = os.path.join(data_dir, "ProsumerHourly_withUTC.csv")
        weather_path = os.path.join(data_dir, "WeatherData.csv")

        # Load & Clean
        df_load = pd.read_csv(load_path)
        df_weather = pd.read_csv(weather_path)

        for df, time_col in [(df_load, "TimeUTC"), (df_weather, "HourUTC")]:
            df["HourUTC"] = pd.to_datetime(df[time_col])
            df.drop(columns=[time_col] if time_col != "HourUTC" else [], inplace=True)
        
        # Merge and Filter
        df = pd.merge(df_load, df_weather, on="HourUTC")
        df = df[df["HourUTC"].dt.year.isin(self.config["data_years"])]
        df = df.rename(columns={"Consumption": "Load"}).dropna()
        
        # Feature Engineering
        df = self._add_cyclical_features(df)
        
        # Set Index for ARIMA
        df = df.set_index("HourUTC").sort_index()
        return df

    def _add_cyclical_features(self, df):
        time_map = {
            "hour": (df["HourUTC"].dt.hour, 24),
            "dow": (df["HourUTC"].dt.dayofweek, 7),
            "month": (df["HourUTC"].dt.month, 12)
        }
        for prefix, (series, max_val) in time_map.items():
            df[f"{prefix}_sin"] = np.sin(2 * np.pi * series / max_val)
            df[f"{prefix}_cos"] = np.cos(2 * np.pi * series / max_val)
        return df

    def split_and_scale(self, df):
        y = df["PV"]
        exog = df[self.exog_cols]
        
        split_idx = int(len(df) * self.config["train_ratio"])
        
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        exog_train, exog_test = exog.iloc[:split_idx], exog.iloc[split_idx:]
        
        # Scale Exogenous features
        exog_train_scaled = pd.DataFrame(self.scaler.fit_transform(exog_train), 
                                         index=exog_train.index, columns=self.exog_cols)
        exog_test_scaled = pd.DataFrame(self.scaler.transform(exog_test), 
                                        index=exog_test.index, columns=self.exog_cols)
        
        return y_train, y_test, exog_train_scaled, exog_test_scaled

# -------------------------------
# 3. ARIMA Forecaster Module
# -------------------------------
class ARIMAForecaster:
    def __init__(self, config):
        self.config = config
        self.model = None

    def fit(self, y_train, exog_train):
        print("Searching for optimal SARIMAX parameters...")
        self.model = pm.auto_arima(
            y=y_train, X=exog_train,
            seasonal=True, m=self.config["seasonal_period"],
            stepwise=True, suppress_warnings=True, max_p=3, max_q=3
        )
        print(self.model.summary())
        
    def save(self, path):
        if self.model is not None:
            joblib.dump(self.model, path)
            print(f"Model successfully saved to {path}")

    def load(self, path):
        self.model = joblib.load(path)
        print(f"Model successfully loaded from {path}")

    def rolling_forecast(self, y_test, exog_test):
        predictions = []
        window = self.config["rolling_window"]
        
        print(f"Starting rolling forecast (updating every {window} steps)...")
        for i in range(0, len(y_test), window):
            end_idx = i + window
            curr_exog = exog_test.iloc[i:end_idx]
            
            # Predict
            chunk_preds = self.model.predict(n_periods=len(curr_exog), X=curr_exog)
            predictions.extend(chunk_preds)
            
            # Update model with actuals
            self.model.update(y_test.iloc[i:end_idx], curr_exog)
            
        return pd.Series(predictions, index=y_test.index)

# -------------------------------
# 4. Main Execution
# -------------------------------
if __name__ == "__main__":
    # Setup Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Check your data path! Not found at: {data_dir}")

    # Process Data
    processor = DataProcessor(CONFIG)
    full_df = processor.load_and_process(data_dir)
    y_train, y_test, X_train, X_test = processor.split_and_scale(full_df)

    # Train and Forecast
    forecaster = ARIMAForecaster(CONFIG)
    forecaster.fit(y_train, X_train)
    
    # Run Rolling Forecast
    y_preds = forecaster.rolling_forecast(y_test, X_test)

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index[:168], y_test.values[:168], label="Actual PV", alpha=0.6)
    plt.plot(y_test.index[:168], y_preds[:168], label="Rolling ARIMA", color='red', linestyle="--")
    plt.title("Solar PV Forecasting: Rolling SARIMAX")
    plt.ylabel("kW")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    import joblib

    # To save
    model_path = os.path.join(base_dir, "sarimax_model.pkl")
    forecaster.save(model_path)