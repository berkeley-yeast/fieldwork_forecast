#!/usr/bin/env python3
"""
Fieldwork Delivery Forecasting Script
Uses skforecast with LightGBM for complex seasonality and exogenous trends
Integrates with Google Sheets API for data input/output
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Google Sheets API
import gspread
from google.oauth2.service_account import Credentials

# Forecasting libraries
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import grid_search_forecaster
import lightgbm as lgb

# Data processing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Visualization
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for GitHub Actions
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Set style safely
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

class FieldworkForecaster:
    def __init__(self, google_creds_path=None):
        """Initialize the forecaster with Google Sheets credentials"""
        self.sheet_name = "large_account_na_data"
        self.tab_name = "Fieldwork"
        self.forecast_horizon = 28  # 4 weeks (more reasonable for stable predictions)
        self.today = datetime.now().date()
        
        # Google Sheets setup
        if google_creds_path:
            self.setup_google_sheets(google_creds_path)
        else:
            # For GitHub Actions, use environment variable
            creds_json = os.environ.get('GOOGLE_SHEETS_CREDENTIALS')
            if creds_json:
                creds_dict = json.loads(creds_json)
                self.setup_google_sheets_from_dict(creds_dict)
            else:
                raise ValueError("Google Sheets credentials not found")
    
    def setup_google_sheets(self, creds_path):
        """Setup Google Sheets API with service account credentials file"""
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        
        creds = Credentials.from_service_account_file(creds_path, scopes=scope)
        self.gc = gspread.authorize(creds)
        
    def setup_google_sheets_from_dict(self, creds_dict):
        """Setup Google Sheets API with service account credentials from dict"""
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        self.gc = gspread.authorize(creds)
    
    def load_data_from_sheets(self):
        """Load data from Google Sheets"""
        print(f"Loading data from Google Sheet: {self.sheet_name}, Tab: {self.tab_name}")
        
        try:
            # Open the spreadsheet and worksheet
            sheet = self.gc.open(self.sheet_name)
            worksheet = sheet.worksheet(self.tab_name)
            
            # Get all values with expected headers to handle duplicates
            try:
                data = worksheet.get_all_records()
            except gspread.exceptions.GSpreadException as e:
                if "duplicates" in str(e):
                    print("Handling duplicate headers in worksheet...")
                    # Get raw data and create our own headers
                    all_values = worksheet.get_all_values()
                    if not all_values:
                        raise ValueError("No data found in worksheet")
                    
                    # Use first row as headers, but make them unique
                    headers = all_values[0]
                    unique_headers = []
                    header_counts = {}
                    
                    for header in headers:
                        if header in header_counts:
                            header_counts[header] += 1
                            unique_headers.append(f"{header}_{header_counts[header]}")
                        else:
                            header_counts[header] = 0
                            unique_headers.append(header)
                    
                    # Create dataframe manually
                    data_rows = all_values[1:]  # Skip header row
                    data = []
                    for row in data_rows:
                        # Pad row with empty strings if shorter than headers
                        padded_row = row + [''] * (len(unique_headers) - len(row))
                        data.append(dict(zip(unique_headers, padded_row)))
                else:
                    raise e
            
            if not data:
                raise ValueError("No data found in the specified sheet/tab")
            
            df = pd.DataFrame(data)
            print(f"Loaded {len(df)} rows of data")
            print(f"Columns: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            print(f"Error loading data from Google Sheets: {str(e)}")
            raise
    
    def preprocess_data(self, df):
        """Preprocess the data for forecasting"""
        print("Preprocessing data...")
        
        # Find the required columns (they might have different names or duplicates)
        print(f"Available columns: {df.columns.tolist()}")
        
        # Try to find date column
        date_col = None
        for col in df.columns:
            if 'date' in col.lower():
                date_col = col
                break
        
        # Try to find units column  
        units_col = None
        for col in df.columns:
            if 'by_units' in col.lower() or 'units' in col.lower():
                units_col = col
                break
        
        # Try to find strain column
        strain_col = None
        for col in df.columns:
            if 'strain' in col.lower() and 'type' not in col.lower():
                strain_col = col
                break
        
        # Try to find strain_type column
        strain_type_col = None
        for col in df.columns:
            if 'strain_type' in col.lower():
                strain_type_col = col
                break
        
        if not date_col:
            raise ValueError(f"Could not find date column in: {df.columns.tolist()}")
        if not units_col:
            raise ValueError(f"Could not find units column in: {df.columns.tolist()}")
        if not strain_col:
            raise ValueError(f"Could not find strain column in: {df.columns.tolist()}")
        if not strain_type_col:
            raise ValueError(f"Could not find strain_type column in: {df.columns.tolist()}")
        
        print(f"Using columns - Date: {date_col}, Units: {units_col}, Strain: {strain_col}, Strain Type: {strain_type_col}")
        
        # Rename columns to standard names
        df = df.rename(columns={
            date_col: 'date',
            units_col: 'by_units', 
            strain_col: 'strain',
            strain_type_col: 'strain_type'
        })
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['date'])
        
        # Convert by_units to numeric
        df['by_units'] = pd.to_numeric(df['by_units'], errors='coerce')
        
        # Remove rows with invalid units
        df = df.dropna(subset=['by_units'])
        
        # Sort by date
        df = df.sort_values('date')
        
        # Create strain encoding for exogenous variables
        self.strain_encoder = LabelEncoder()
        df['strain_encoded'] = self.strain_encoder.fit_transform(df['strain'].astype(str))
        
        # Create strain_type encoding for exogenous variables
        self.strain_type_encoder = LabelEncoder()
        df['strain_type_encoded'] = self.strain_type_encoder.fit_transform(df['strain_type'].astype(str))
        
        print(f"Data preprocessed: {len(df)} valid rows")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Unique strains: {df['strain'].nunique()}")
        print(f"Unique strain types: {df['strain_type'].nunique()}")
        
        return df
    
    def create_time_series(self, df):
        """Create time series with proper frequency and exogenous variables"""
        print("Creating time series...")
        
        # Set date as index, but first handle any duplicate dates
        df = df.set_index('date')
        
        # Remove duplicate indices by keeping the first occurrence
        if df.index.duplicated().any():
            print(f"Found {df.index.duplicated().sum()} duplicate dates, keeping first occurrence")
            df = df[~df.index.duplicated(keep='first')]
        
        # Aggregate by date (sum units, mode for strain and strain_type)
        daily_data = df.groupby(df.index.date).agg({
            'by_units': 'sum',
            'strain_encoded': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0,
            'strain_type_encoded': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0
        })
        
        # Create full date range
        full_range = pd.date_range(
            start=daily_data.index.min(), 
            end=daily_data.index.max(), 
            freq='D'
        )
        
        # Reindex to fill missing dates
        daily_data = daily_data.reindex(full_range, fill_value=0)
        
        # Add comprehensive time-based features for seasonality
        daily_data['day_of_week'] = daily_data.index.dayofweek
        daily_data['day_of_month'] = daily_data.index.day
        daily_data['month'] = daily_data.index.month
        daily_data['quarter'] = daily_data.index.quarter
        
        # Add holiday and special date features
        daily_data['is_month_end'] = (daily_data.index.day >= 28).astype(int)
        daily_data['is_month_start'] = (daily_data.index.day <= 3).astype(int)
        daily_data['is_quarter_end'] = daily_data.index.to_series().apply(
            lambda x: 1 if x.month in [3, 6, 9, 12] and x.day >= 28 else 0
        ).values
        
        # Add days since last delivery feature (trend)
        last_delivery_idx = daily_data[daily_data['by_units'] > 0].index
        if len(last_delivery_idx) > 0:
            daily_data['days_since_last_delivery'] = 0
            for i, date in enumerate(daily_data.index):
                if len(last_delivery_idx[last_delivery_idx <= date]) > 0:
                    last_delivery = last_delivery_idx[last_delivery_idx <= date][-1]
                    daily_data.loc[date, 'days_since_last_delivery'] = (date - last_delivery).days
                else:
                    daily_data.loc[date, 'days_since_last_delivery'] = 999  # No prior delivery
        else:
            daily_data['days_since_last_delivery'] = 999
        
        # Add rolling averages (but not manual lags since ForecasterRecursive handles those)
        daily_data['rolling_7'] = daily_data['by_units'].rolling(7).mean()
        daily_data['rolling_14'] = daily_data['by_units'].rolling(14).mean()
        
        # Drop initial NaN values from rolling averages
        daily_data = daily_data.dropna()
        
        print(f"Time series created with {len(daily_data)} daily observations")
        
        return daily_data
    
    def train_intermittent_forecaster(self, ts_data):
        """Train an intermittent demand forecasting model"""
        print("Training intermittent demand forecaster...")
        
        # Intermittent forecasting approach:
        # 1. Model probability of delivery occurring (binary classification)
        # 2. Model delivery size when it occurs (regression on non-zero values)
        
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegression
        
        # Prepare comprehensive features for intermittent forecasting
        exog_vars = [
            'strain_encoded', 'strain_type_encoded', 'day_of_week', 'day_of_month', 
            'month', 'quarter', 'rolling_7', 'rolling_14',
            'is_month_end', 'is_month_start', 'is_quarter_end',
            'days_since_last_delivery'
        ]
        
        # Store the training feature order for later use in prediction
        self.training_features = exog_vars
        
        # Create binary target (delivery vs no delivery)
        ts_data['has_delivery'] = (ts_data['by_units'] > 0).astype(int)
        
        # Model 1: Probability of delivery occurring
        print("Training delivery probability model...")
        self.delivery_probability_model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        
        # Model 2: Delivery size when it occurs
        print("Training delivery size model...")
        delivery_data = ts_data[ts_data['by_units'] > 0].copy()
        
        if len(delivery_data) > 0:
            self.delivery_size_model = RandomForestRegressor(
                n_estimators=30,
                max_depth=3,
                min_samples_split=5,
                random_state=42
            )
            
            # Check available features in training data
            print(f"Available features in ts_data: {list(ts_data.columns)}")
            print(f"Required features: {exog_vars}")
            
            # Check for missing features in training data
            missing_in_training = set(exog_vars) - set(ts_data.columns)
            if missing_in_training:
                print(f"Missing features in training data: {missing_in_training}")
                # Remove missing features from the list
                exog_vars = [f for f in exog_vars if f in ts_data.columns]
                self.training_features = exog_vars
                print(f"Updated feature list: {exog_vars}")
            
            # Fit probability model on all data
            X_prob = ts_data[exog_vars]
            y_prob = ts_data['has_delivery']
            self.delivery_probability_model.fit(X_prob, y_prob)
            
            # Fit size model on delivery days only
            X_size = delivery_data[exog_vars]
            y_size = delivery_data['by_units']
            self.delivery_size_model.fit(X_size, y_size)
            
            print(f"Models trained on {len(ts_data)} total days, {len(delivery_data)} delivery days")
            print(f"Delivery probability: {len(delivery_data)/len(ts_data)*100:.1f}% of days")
            
        else:
            print("No delivery data found for training size model")
            self.delivery_size_model = None
        
        return ts_data
    
    def make_forecast(self, ts_data):
        """Generate forecast for the next 4 weeks from today's date"""
        print(f"Generating {self.forecast_horizon}-day forecast from today ({self.today})...")
        
        # Calculate forecast dates starting from tomorrow (not today)
        tomorrow = self.today + timedelta(days=1)
        forecast_start = datetime.combine(tomorrow, datetime.min.time())
        forecast_dates = pd.date_range(
            start=forecast_start,
            periods=self.forecast_horizon,
            freq='D'
        )
        
        # Calculate how many days to forecast beyond the training data
        last_data_date = ts_data.index[-1]
        days_gap = (forecast_start.date() - last_data_date.date()).days
        total_steps = self.forecast_horizon + max(0, days_gap)
        
        print(f"Last data date: {last_data_date.date()}")
        print(f"Forecast start date: {forecast_start.date()}")
        print(f"Days gap: {days_gap}, Total steps to forecast: {total_steps}")
        
        # Create exogenous variables for forecast period
        exog_future = pd.DataFrame(index=forecast_dates)
        
        # Use last known strain and strain_type as default
        last_strain = ts_data['strain_encoded'].iloc[-1]
        last_strain_type = ts_data['strain_type_encoded'].iloc[-1]
        exog_future['strain_encoded'] = last_strain
        exog_future['strain_type_encoded'] = last_strain_type
        
        # Time-based features
        exog_future['day_of_week'] = exog_future.index.dayofweek
        exog_future['day_of_month'] = exog_future.index.day
        exog_future['month'] = exog_future.index.month
        exog_future['quarter'] = exog_future.index.quarter
        
        # For rolling features, use recent averages as proxies
        recent_avg_7 = ts_data['by_units'].tail(7).mean()
        recent_avg_14 = ts_data['by_units'].tail(14).mean()
        
        exog_future['rolling_7'] = recent_avg_7
        exog_future['rolling_14'] = recent_avg_14
        
        # Create exogenous variables for forecast period
        forecast_exog = pd.DataFrame(index=forecast_dates)
        
        # Use last known strain and strain_type as default
        last_strain = ts_data['strain_encoded'].iloc[-1]
        last_strain_type = ts_data['strain_type_encoded'].iloc[-1]
        forecast_exog['strain_encoded'] = last_strain
        forecast_exog['strain_type_encoded'] = last_strain_type
        
        # Time-based features
        forecast_exog['day_of_week'] = forecast_exog.index.dayofweek
        forecast_exog['day_of_month'] = forecast_exog.index.day
        forecast_exog['month'] = forecast_exog.index.month
        forecast_exog['quarter'] = forecast_exog.index.quarter
        
        # Special date features
        forecast_exog['is_month_end'] = (forecast_exog.index.day >= 28).astype(int)
        forecast_exog['is_month_start'] = (forecast_exog.index.day <= 3).astype(int)
        forecast_exog['is_quarter_end'] = forecast_exog.index.to_series().apply(
            lambda x: 1 if x.month in [3, 6, 9, 12] and x.day >= 28 else 0
        ).values
        
        # Days since last delivery (critical for intermittent forecasting)
        last_delivery_date = ts_data[ts_data['by_units'] > 0].index[-1] if len(ts_data[ts_data['by_units'] > 0]) > 0 else ts_data.index[0]
        forecast_exog['days_since_last_delivery'] = [(date - last_delivery_date).days for date in forecast_exog.index]
        
        # For rolling features, use recent averages as proxies
        recent_avg_7 = ts_data['by_units'].tail(7).mean()
        recent_avg_14 = ts_data['by_units'].tail(14).mean()
        forecast_exog['rolling_7'] = recent_avg_7
        forecast_exog['rolling_14'] = recent_avg_14

        # Generate intermittent forecast
        print("Generating intermittent demand forecast...")
        
        # Ensure forecast features match training features exactly
        print(f"Training features: {self.training_features}")
        print(f"Forecast features: {list(forecast_exog.columns)}")
        
        # Check for missing features
        missing_features = set(self.training_features) - set(forecast_exog.columns)
        if missing_features:
            print(f"Missing features in forecast data: {missing_features}")
            raise ValueError(f"Missing features: {missing_features}")
        
        # Reorder forecast features to match training order
        forecast_exog = forecast_exog[self.training_features]
        
        # Step 1: Predict probability of delivery for each day
        delivery_probabilities = self.delivery_probability_model.predict_proba(forecast_exog)[:, 1]
        
        # Step 2: Predict delivery sizes
        if self.delivery_size_model is not None:
            delivery_sizes = self.delivery_size_model.predict(forecast_exog)
            delivery_sizes = np.maximum(delivery_sizes, 0)  # Ensure non-negative
        else:
            # Fallback: use historical average
            historical_avg = ts_data[ts_data['by_units'] > 0]['by_units'].mean()
            delivery_sizes = np.full(len(forecast_dates), historical_avg)
        
        # Step 3: Combine probability and size to get expected delivery
        # Use a more conservative threshold for intermittent deliveries
        probability_threshold = 0.1  # Lower threshold but still selective
        
        forecast_values = []
        for i, (prob, size) in enumerate(zip(delivery_probabilities, delivery_sizes)):
            if prob > probability_threshold:
                # Use actual predicted size, not scaled by probability
                forecast_values.append(size)
            else:
                forecast_values.append(0)
        
        forecast = np.array(forecast_values)
        
        print(f"Intermittent forecast generated:")
        print(f"• Days with delivery probability > {probability_threshold}: {np.sum(forecast > 0)}")
        print(f"• Average delivery probability: {np.mean(delivery_probabilities):.3f}")
        print(f"• Max predicted delivery: {np.max(forecast):.1f} BBLs")
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'predicted_units': forecast,
            'delivery_probability': delivery_probabilities
        })
        
        # Light smoothing only (intermittent forecasting is naturally less spiky)
        forecast_df['predicted_units'] = forecast_df['predicted_units'].clip(lower=0, upper=200)
        
        # Find the next predicted delivery date (any amount > 0, not just >= 100 BBLs)
        next_any_delivery = forecast_df[forecast_df['predicted_units'] > 0]
        if len(next_any_delivery) > 0:
            next_any_delivery_date = next_any_delivery['date'].iloc[0]
            next_any_delivery_amount = next_any_delivery['predicted_units'].iloc[0]
        else:
            next_any_delivery_date = None
            next_any_delivery_amount = 0
        
        # Apply pattern-based prediction overlay
        # Look at historical delivery patterns to enhance predictions
        historical_delivery_dates = ts_data[ts_data['by_units'] >= 100.0].index
        pattern_based_next_delivery = None
        
        if len(historical_delivery_dates) >= 2:
            # Calculate average days between significant deliveries
            intervals = [(historical_delivery_dates[i] - historical_delivery_dates[i-1]).days 
                        for i in range(1, len(historical_delivery_dates))]
            avg_interval = sum(intervals) / len(intervals)
            
            # Predict next delivery based on pattern
            last_delivery = historical_delivery_dates[-1]
            pattern_based_next_delivery = last_delivery + timedelta(days=int(avg_interval))
            
            print(f"📊 Pattern-based analysis:")
            print(f"• Average interval between deliveries: {avg_interval:.1f} days")
            print(f"• Last delivery: {last_delivery.date()}")
            print(f"• Expected next delivery (pattern-based): {pattern_based_next_delivery.date()}")
            
            # Boost prediction around expected delivery date
            for i, row in forecast_df.iterrows():
                days_from_expected = abs((row['date'] - pattern_based_next_delivery).days)
                if days_from_expected <= 3:  # Within 3 days of expected
                    boost_factor = 1.5 - (days_from_expected * 0.1)
                    forecast_df.loc[i, 'predicted_units'] *= boost_factor
                    print(f"📈 Boosted prediction for {row['date'].date()}: {forecast_df.loc[i, 'predicted_units']:.1f} BBLs")
        
        # For intermittent deliveries, find the most confident prediction and filter accordingly
        all_predictions = forecast_df[forecast_df['predicted_units'] > 0]
        
        if len(all_predictions) > 0:
            # Find the prediction with highest confidence (highest predicted units)
            most_confident = all_predictions.loc[all_predictions['predicted_units'].idxmax()]
            most_confident_date = most_confident['date']
            
            print(f"Most confident prediction: {most_confident_date.date()} with {most_confident['predicted_units']:.1f} BBLs")
            
            # Only show predictions on or after the most confident date
            # This removes earlier, less confident predictions
            top_forecasts = all_predictions[all_predictions['date'] >= most_confident_date].copy()
            top_forecasts = top_forecasts.sort_values('date')  # Sort by date
            
            print(f"Filtered to {len(top_forecasts)} predictions on/after most confident date")
        else:
            top_forecasts = pd.DataFrame()
        
        if len(top_forecasts) > 0:
            print(f"Top {len(top_forecasts)} predicted deliveries:")
            for i, (_, row) in enumerate(top_forecasts.iterrows(), 1):
                print(f"  {i}. {row['date'].date()}: {row['predicted_units']:.1f} BBLs")
            
            # Use the highest probability delivery as best estimate
            best_next_delivery_date = top_forecasts['date'].iloc[0]
            print(f"🎯 Next delivery (ML): {best_next_delivery_date.date()}")
        elif pattern_based_next_delivery is not None:
            # Use pattern-based prediction
            best_next_delivery_date = pattern_based_next_delivery
            print(f"🎯 Next delivery (pattern-based): {best_next_delivery_date.date()}")
            
            # Don't include pattern-based predictions in the main forecast
            # (they'll be handled separately in the summary)
            top_forecasts = pd.DataFrame()
        else:
            best_next_delivery_date = None
            top_forecasts = pd.DataFrame()
            print("🎯 No next delivery date could be determined")
        
        forecast_df = top_forecasts
        
        # Use the earliest forecast date as the next delivery date (should match the forecast)
        if len(forecast_df) > 0:
            next_delivery_date = forecast_df['date'].iloc[0]
            print(f"Next predicted delivery date: {next_delivery_date.date()}")
            print(f"Predicted units: {forecast_df['predicted_units'].iloc[0]:.1f}")
            
            # Use the ML prediction as the best estimate (overriding pattern-based)
            best_next_delivery_date = next_delivery_date
        else:
            print("No deliveries predicted in the next 4 weeks")
            next_delivery_date = best_next_delivery_date  # Use pattern-based if available
        
        # Return both the significant forecast and the best next delivery date estimate
        return forecast_df, next_delivery_date, best_next_delivery_date
    
    def calculate_delivery_statistics(self, ts_data):
        """Calculate statistics about delivery patterns"""
        print("Calculating delivery statistics...")
        
        # Find actual delivery dates (where units > 0)
        delivery_dates = ts_data[ts_data['by_units'] > 0].index.tolist()
        
        if len(delivery_dates) < 2:
            print("Not enough delivery dates to calculate statistics")
            return None
        
        # Calculate days between consecutive deliveries
        days_between = []
        for i in range(1, len(delivery_dates)):
            days_diff = (delivery_dates[i] - delivery_dates[i-1]).days
            days_between.append(days_diff)
        
        if not days_between:
            return None
        
        # Calculate statistics
        avg_days_between = sum(days_between) / len(days_between)
        min_days_between = min(days_between)
        max_days_between = max(days_between)
        
        # Calculate statistics for significant deliveries (≥100 BBLs)
        significant_delivery_dates = ts_data[ts_data['by_units'] >= 100.0].index.tolist()
        
        avg_days_between_significant = None
        if len(significant_delivery_dates) >= 2:
            sig_days_between = []
            for i in range(1, len(significant_delivery_dates)):
                days_diff = (significant_delivery_dates[i] - significant_delivery_dates[i-1]).days
                sig_days_between.append(days_diff)
            
            if sig_days_between:
                avg_days_between_significant = sum(sig_days_between) / len(sig_days_between)
        
        stats = {
            'avg_days_between_all': avg_days_between,
            'min_days_between': min_days_between,
            'max_days_between': max_days_between,
            'avg_days_between_significant': avg_days_between_significant,
            'total_deliveries': len(delivery_dates),
            'significant_deliveries': len(significant_delivery_dates)
        }
        
        print(f"Average days between deliveries (all): {avg_days_between:.1f}")
        print(f"Average days between significant deliveries (≥100 BBLs): {avg_days_between_significant:.1f}" if avg_days_between_significant else "Not enough significant deliveries")
        print(f"Range: {min_days_between} to {max_days_between} days")
        
        # Analyze the delivery pattern to understand spikiness
        delivery_volumes = ts_data[ts_data['by_units'] > 0]['by_units'].values
        zero_days = len(ts_data[ts_data['by_units'] == 0])
        delivery_days = len(ts_data[ts_data['by_units'] > 0])
        
        print(f"\n📊 Delivery Pattern Analysis:")
        print(f"• Zero delivery days: {zero_days}")
        print(f"• Delivery days: {delivery_days}")
        print(f"• Delivery frequency: {delivery_days/(delivery_days + zero_days)*100:.1f}% of days")
        print(f"• Average delivery size: {delivery_volumes.mean():.1f} BBLs")
        print(f"• This explains the spiky forecast - deliveries are rare but large!")
        
        return stats
    
    def create_forecast_visualization(self, ts_data, forecast_df, next_delivery_date):
        """Create visualization showing historical data, model fit, and forecast"""
        print("Creating forecast visualization...")
        
        try:
            # Set up the plot
            plt.figure(figsize=(15, 10))
            
            # Prepare data for plotting - extend to show 2 weeks from today
            last_60_days = ts_data.tail(60).copy()  # Show last 60 days of historical data
            
            # Create the full forecast period (all predictions, not just ≥100 BBLs)
            forecast_start = datetime.combine(self.today, datetime.min.time())
            full_forecast_dates = pd.date_range(
                start=forecast_start,
                periods=self.forecast_horizon,
                freq='D'
            )
            
            # Generate the full forecast for visualization (including all values)
            try:
                last_data_date = ts_data.index[-1]
                days_gap = (forecast_start.date() - last_data_date.date()).days
                total_steps = self.forecast_horizon + max(0, days_gap)
                
                print(f"Visualization: Last data date: {last_data_date.date()}")
                print(f"Visualization: Forecast start: {forecast_start.date()}")
                print(f"Visualization: Days gap: {days_gap}, Total steps: {total_steps}")
                
                # We need to create exogenous variables for the full forecast period
                if days_gap > 0:
                    # Create extended exogenous variables for the full forecast period
                    extended_dates = pd.date_range(
                        start=last_data_date + timedelta(days=1),
                        periods=total_steps,
                        freq='D'
                    )
                    extended_exog = pd.DataFrame(index=extended_dates)
                    
                    # Use last known strain and strain_type as default
                    last_strain = ts_data['strain_encoded'].iloc[-1]
                    last_strain_type = ts_data['strain_type_encoded'].iloc[-1]
                    extended_exog['strain_encoded'] = last_strain
                    extended_exog['strain_type_encoded'] = last_strain_type
                    
                    # Time-based features
                    extended_exog['day_of_week'] = extended_exog.index.dayofweek
                    extended_exog['day_of_month'] = extended_exog.index.day
                    extended_exog['month'] = extended_exog.index.month
                    extended_exog['quarter'] = extended_exog.index.quarter
                    
                    # Rolling averages
                    recent_avg_7 = ts_data['by_units'].tail(7).mean()
                    recent_avg_14 = ts_data['by_units'].tail(14).mean()
                    extended_exog['rolling_7'] = recent_avg_7
                    extended_exog['rolling_14'] = recent_avg_14
                    
                    viz_exog = extended_exog
                else:
                    # Create exog for just the forecast period
                    viz_exog = pd.DataFrame(index=full_forecast_dates)
                    last_strain = ts_data['strain_encoded'].iloc[-1]
                    last_strain_type = ts_data['strain_type_encoded'].iloc[-1]
                    viz_exog['strain_encoded'] = last_strain
                    viz_exog['strain_type_encoded'] = last_strain_type
                    viz_exog['day_of_week'] = viz_exog.index.dayofweek
                    viz_exog['day_of_month'] = viz_exog.index.day
                    viz_exog['month'] = viz_exog.index.month
                    viz_exog['quarter'] = viz_exog.index.quarter
                    recent_avg_7 = ts_data['by_units'].tail(7).mean()
                    recent_avg_14 = ts_data['by_units'].tail(14).mean()
                    viz_exog['rolling_7'] = recent_avg_7
                    viz_exog['rolling_14'] = recent_avg_14
                
                # Generate forecast for visualization
                full_viz_forecast = self.forecaster.predict(steps=total_steps, exog=viz_exog)
                
                if days_gap > 0:
                    viz_forecast_values = full_viz_forecast[-self.forecast_horizon:]
                else:
                    viz_forecast_values = full_viz_forecast
                    
                # Create full forecast dataframe for visualization
                full_forecast_df = pd.DataFrame({
                    'date': full_forecast_dates,
                    'predicted_units': viz_forecast_values
                })
                
                print(f"Generated visualization forecast with {len(full_forecast_df)} points")
                
            except Exception as e:
                print(f"Error generating visualization forecast: {str(e)}")
                # Fallback: create dummy forecast data for visualization
                full_forecast_df = pd.DataFrame({
                    'date': full_forecast_dates,
                    'predicted_units': [0] * len(full_forecast_dates)
                })
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
            
            # Plot 1: Historical data with continuous forecast line
            ax1.plot(last_60_days.index, last_60_days['by_units'], 
                    'b-', linewidth=2, label='Historical Deliveries', alpha=0.7)
            
            # Create a complete continuous line from historical data through forecast
            if len(full_forecast_df) > 0:
                # Get the last few historical points to create smooth transition
                transition_historical = last_60_days.tail(5)  # Last 5 days for smooth connection
                
                # Combine historical transition data with forecast data
                combined_dates = list(transition_historical.index) + list(full_forecast_df['date'])
                combined_values = list(transition_historical['by_units']) + list(full_forecast_df['predicted_units'])
                
                # Plot the complete continuous line
                ax1.plot(combined_dates, combined_values, 
                        'b-', linewidth=2, label='Complete Model (Historical + Forecast)', alpha=0.8)
                
                # Add a vertical line to separate historical from forecast
                forecast_start_line = full_forecast_df['date'].iloc[0]
                ax1.axvline(x=forecast_start_line, color='orange', linestyle='--', alpha=0.6, linewidth=1, label='Forecast Start')
            
            # Add significant forecast points (≥100 BBLs) as markers
            # Size markers by delivery probability for intermittent forecasting
            if len(forecast_df) > 0:
                # Get full forecast data for visualization (including probabilities)
                full_forecast_viz = pd.DataFrame({
                    'date': full_forecast_df['date'],
                    'predicted_units': full_forecast_df['predicted_units']
                })
                
                # Add probability info if available
                if 'delivery_probability' in full_forecast_df.columns:
                    full_forecast_viz['probability'] = full_forecast_df['delivery_probability']
                    # Size scatter points by probability
                    sizes = full_forecast_viz['probability'] * 100  # Scale for visibility
                    ax1.scatter(forecast_df['date'], forecast_df['predicted_units'], 
                               c=forecast_df['delivery_probability'], s=sizes, 
                               cmap='Reds', label='Deliveries (size=probability)', 
                               marker='o', zorder=5, edgecolor='darkred', alpha=0.7)
                else:
                    ax1.scatter(forecast_df['date'], forecast_df['predicted_units'], 
                               color='red', s=50, label='Significant Deliveries (≥100 BBLs)', 
                               marker='o', zorder=5, edgecolor='darkred')
                
                # Highlight next delivery with larger marker
                if next_delivery_date is not None:
                    next_delivery_units = forecast_df[forecast_df['date'] == next_delivery_date]['predicted_units'].iloc[0]
                    ax1.scatter(next_delivery_date, next_delivery_units, 
                               color='red', s=150, label=f'Next Delivery: {next_delivery_date.date()}',
                               marker='*', zorder=6, edgecolor='darkred')
            
            # Add vertical line to show today
            ax1.axvline(x=forecast_start, color='green', linestyle=':', alpha=0.7, linewidth=2, label='Today')
            
            ax1.set_title('Fieldwork Delivery Forecast', fontsize=16, fontweight='bold')
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Delivery Volume (BBLs)', fontsize=12)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Set x-axis limits to ensure full 3-week forecast period is visible
            forecast_end = forecast_start + timedelta(days=self.forecast_horizon)
            ax1.set_xlim(last_60_days.index[0], forecast_end)
            
            # Format x-axis dates
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))  # Every 2 days to avoid crowding
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Plot 2: Distribution of historical deliveries
            ax2.hist(ts_data[ts_data['by_units'] > 0]['by_units'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(100, color='red', linestyle='--', linewidth=2, label='100 BBL Threshold')
            ax2.set_title('Distribution of Historical Delivery Volumes', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Delivery Volume (BBLs)', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add summary statistics
            stats_text = f"""
Model Summary:
• Training Period: {ts_data.index.min().date()} to {ts_data.index.max().date()}
• Total Observations: {len(ts_data)}
• Average Delivery: {ts_data[ts_data['by_units'] > 0]['by_units'].mean():.1f} BBLs
• Forecast Generated: {self.today}
• Next Delivery ≥100 BBLs: {next_delivery_date.date() if next_delivery_date else 'None in next 4 weeks'}
            """
            
            plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)  # Make room for summary text
            
            # Save the plot
            plot_filename = f"fieldwork_forecast_{self.today.strftime('%Y%m%d')}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Forecast visualization saved as: {plot_filename}")
            
            plt.close()  # Close to free memory
            
            return plot_filename
            
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            return None
    
    def write_forecast_to_sheets(self, forecast_df, next_delivery_date, all_next_delivery_dates=None, best_next_delivery_date=None):
        """Write forecast results to Google Sheets starting at column F (index 6)"""
        print("Writing forecast to Google Sheets...")
        
        # Use best next delivery date if available, otherwise use the significant delivery date
        display_next_delivery = best_next_delivery_date if best_next_delivery_date is not None else next_delivery_date
        
        try:
            sheet = self.gc.open(self.sheet_name)
            worksheet = sheet.worksheet(self.tab_name)
            
            # Clear existing forecast data completely (columns F onwards)
            # Get the current sheet dimensions
            all_values = worksheet.get_all_values()
            if all_values:
                last_row = len(all_values)
                # Clear a wide range to ensure all previous forecast data is removed
                # Columns F through Z should be more than enough
                clear_range = f"F1:Z{last_row + 10}"  # Add extra rows for safety
                print(f"Clearing previous forecast data in range: {clear_range}")
                worksheet.batch_clear([clear_range])
                
                # Also clear any data that might be in columns beyond Z
                extended_clear_range = f"AA1:AZ{last_row + 10}"
                try:
                    worksheet.batch_clear([extended_clear_range])
                    print("Also cleared extended range AA:AZ")
                except:
                    pass  # Ignore errors if these columns don't exist
                
                # Small delay to ensure clearing is complete
                import time
                time.sleep(1)
                
                # Verify clearing worked by checking if F1 is empty
                try:
                    test_cell = worksheet.acell('F1').value
                    if test_cell:
                        print(f"Warning: F1 still contains data: {test_cell}")
                    else:
                        print("Verification: Previous forecast data cleared successfully")
                except:
                    print("Previous forecast data cleared successfully")
            
            # Prepare forecast data for writing
            if next_delivery_date is not None:
                # Write headers with group name column
                headers = ['Forecast Date', 'Predicted Units', 'Group Name', 'Generated On', 'Next Delivery Date']
                worksheet.update('F1:J1', [headers])
                
                # Prepare forecast data
                # Calculate the most likely delivery date for each group
                # Use the prediction with highest confidence (highest predicted units)
                group_best = {}
                for group_name in forecast_df['group_name'].unique():
                    group_data = forecast_df[forecast_df['group_name'] == group_name]
                    # Get the date with the highest predicted units (most confident prediction)
                    best_prediction = group_data.loc[group_data['predicted_units'].idxmax()]
                    group_best[group_name] = best_prediction['date']
                
                forecast_data = []
                for _, row in forecast_df.iterrows():
                    group_name = row.get('group_name', 'Unknown')
                    best_for_group = group_best.get(group_name, row['date'])
                    
                    forecast_data.append([
                        row['date'].strftime('%Y-%m-%d'),
                        f"{row['predicted_units']:.1f}",
                        group_name,
                        self.today.strftime('%Y-%m-%d'),
                        best_for_group.strftime('%Y-%m-%d')  # Show most confident delivery for this group
                    ])
                
                # Write forecast data
                if forecast_data:
                    end_row = len(forecast_data) + 1
                    range_name = f"F2:J{end_row}"
                    worksheet.update(range_name, forecast_data)
                
                # Write summary information by group
                # Collect next delivery dates by group from the combined forecast
                group_summaries = {}
                if len(forecast_df) > 0:
                    # Group by group_name and get the most confident delivery for each
                    for group_name in forecast_df['group_name'].unique():
                        group_data = forecast_df[forecast_df['group_name'] == group_name]
                        # Get the prediction with highest confidence (highest predicted units)
                        best_prediction = group_data.loc[group_data['predicted_units'].idxmax()]
                        best_date = best_prediction['date']
                        days_until = (best_date.date() - self.today).days
                        group_summaries[group_name] = {
                            'date': best_date,
                            'days_until': days_until
                        }
                
                # Also include pattern-based predictions from all_next_delivery_dates
                # that might not be in the main forecast
                if all_next_delivery_dates:
                    for delivery_info in all_next_delivery_dates:
                        group_name = delivery_info['group_name']
                        if group_name not in group_summaries:
                            delivery_date = delivery_info['date']
                            days_until = (delivery_date.date() - self.today).days
                            group_summaries[group_name] = {
                                'date': delivery_date,
                                'days_until': days_until
                            }
                
                # Write group-specific summaries
                summary_headers = ['Group', 'Next Delivery', 'Days Until']
                worksheet.update('K1:M1', [summary_headers])
                
                summary_data = []
                for i, (group_name, info) in enumerate(group_summaries.items()):
                    summary_data.append([
                        group_name,
                        info['date'].strftime('%Y-%m-%d'),
                        str(info['days_until'])
                    ])
                
                if summary_data:
                    end_summary_row = len(summary_data) + 1
                    worksheet.update(f'K2:M{end_summary_row}', summary_data)
                    print(f"✅ Group summaries written to K2:M{end_summary_row}")
                else:
                    # Fallback: write overall next delivery if no group data
                    days_until = (next_delivery_date.date() - self.today).days
                    fallback_data = [['Overall', next_delivery_date.strftime('%Y-%m-%d'), str(days_until)]]
                    worksheet.update('K2:M2', fallback_data)
                
                # Write delivery statistics if available
                if delivery_stats:
                    stats_headers = ['Avg Days Between All Deliveries', 'Avg Days Between Significant Deliveries (≥100 BBLs)', 'Total Deliveries', 'Significant Deliveries']
                    worksheet.update('M1:P1', [stats_headers])
                    
                    stats_data = [
                        f"{delivery_stats['avg_days_between_all']:.1f}",
                        f"{delivery_stats['avg_days_between_significant']:.1f}" if delivery_stats['avg_days_between_significant'] else "N/A",
                        str(delivery_stats['total_deliveries']),
                        str(delivery_stats['significant_deliveries'])
                    ]
                    worksheet.update('M2:P2', [stats_data])
                
                print(f"Forecast written successfully. Next delivery: {next_delivery_date.date()}")
                
            else:
                # No significant deliveries predicted, but still show next delivery date if available
                if display_next_delivery:
                    headers = ['Forecast Status', 'Generated On', 'Next Delivery Date (Any Amount)']
                    worksheet.update('F1:H1', [headers])
                    
                    status_data = ['No deliveries ≥100 BBL predicted in next 4 weeks', 
                                  self.today.strftime('%Y-%m-%d'),
                                  display_next_delivery.strftime('%Y-%m-%d')]
                    worksheet.update('F2:H2', [status_data])
                else:
                    headers = ['Forecast Status', 'Generated On']
                    worksheet.update('F1:G1', [headers])
                    
                    status_data = ['No deliveries predicted in next 4 weeks', self.today.strftime('%Y-%m-%d')]
                    worksheet.update('F2:G2', [status_data])
                
                # Still write delivery statistics if available
                if delivery_stats:
                    stats_headers = ['Avg Days Between All Deliveries', 'Avg Days Between Significant Deliveries (≥100 BBLs)', 'Total Deliveries', 'Significant Deliveries']
                    worksheet.update('J1:M1', [stats_headers])
                    
                    stats_data = [
                        f"{delivery_stats['avg_days_between_all']:.1f}",
                        f"{delivery_stats['avg_days_between_significant']:.1f}" if delivery_stats['avg_days_between_significant'] else "N/A",
                        str(delivery_stats['total_deliveries']),
                        str(delivery_stats['significant_deliveries'])
                    ]
                    worksheet.update('J2:M2', [stats_data])
                
                print("No deliveries predicted - status written to sheet")
                
        except Exception as e:
            print(f"Error writing to Google Sheets: {str(e)}")
            raise
    
    def run_forecast(self):
        """Main method to run the complete forecasting pipeline"""
        print("Starting Fieldwork forecasting pipeline...")
        print(f"Today's date: {self.today}")
        
        try:
            # Load data from Google Sheets
            raw_data = self.load_data_from_sheets()
            
            # Preprocess data
            clean_data = self.preprocess_data(raw_data)
            
            # Check both individual strains and strain types for sufficient data
            strain_candidates = []
            
            # First try individual strains with enough data
            for strain in clean_data['strain'].unique():
                strain_data = clean_data[clean_data['strain'] == strain]
                if len(strain_data) >= 15:  # Higher threshold for individual strains
                    strain_candidates.append(('strain', strain, strain_data))
            
            # If no individual strains have enough data, try strain types
            if not strain_candidates:
                for strain_type in clean_data['strain_type'].unique():
                    strain_type_data = clean_data[clean_data['strain_type'] == strain_type]
                    if len(strain_type_data) >= 10:  # Medium threshold for strain types
                        strain_candidates.append(('strain_type', strain_type, strain_type_data))
            
            print(f"Found {len(strain_candidates)} viable forecasting candidates")
            
            all_forecasts = []
            all_next_delivery_dates = []
            
            # Create separate forecast for each viable strain/strain_type
            for group_type, group_name, strain_data in strain_candidates:
                print(f"\n=== Forecasting for {group_type}: {group_name} ===")
                print(f"Data points: {len(strain_data)} records")
                
                # Create time series for this strain
                ts_data = self.create_time_series(strain_data)
                
                # Train intermittent forecaster for this strain
                self.train_intermittent_forecaster(ts_data)
                
                # Calculate delivery statistics for this strain
                delivery_stats = self.calculate_delivery_statistics(ts_data)
                
                # Make forecast for this strain
                forecast_df, next_delivery_date, best_next_delivery_date = self.make_forecast(ts_data)
                
                # Add group info to forecast
                if len(forecast_df) > 0:
                    forecast_df['group_type'] = group_type
                    forecast_df['group_name'] = group_name
                    all_forecasts.append(forecast_df)
                    
                if next_delivery_date:
                    all_next_delivery_dates.append({
                        'group_type': group_type,
                        'group_name': group_name,
                        'date': next_delivery_date,
                        'best_date': best_next_delivery_date
                    })
                
                # Also add pattern-based predictions even if no ML predictions
                if best_next_delivery_date and not next_delivery_date:
                    all_next_delivery_dates.append({
                        'group_type': group_type,
                        'group_name': group_name,
                        'date': best_next_delivery_date,
                        'best_date': best_next_delivery_date
                    })
            
            # Combine all strain forecasts
            if all_forecasts:
                combined_forecast = pd.concat(all_forecasts, ignore_index=True)
                combined_forecast = combined_forecast.sort_values('date')
                
                # Find the earliest next delivery across all groups
                if all_next_delivery_dates:
                    earliest_delivery = min(all_next_delivery_dates, key=lambda x: x['date'])
                    next_delivery_date = earliest_delivery['date']
                    best_next_delivery_date = earliest_delivery['best_date']
                    print(f"\nEarliest next delivery: {next_delivery_date} ({earliest_delivery['group_name']})")
                else:
                    next_delivery_date = None
                    best_next_delivery_date = None
                
                # Write combined results to Google Sheets
                self.write_forecast_to_sheets(combined_forecast, next_delivery_date, all_next_delivery_dates, best_next_delivery_date)
                
                print("Multi-strain forecasting pipeline completed successfully!")
            else:
                # Even if no ML forecasts, write pattern-based predictions if available
                if all_next_delivery_dates:
                    earliest_delivery = min(all_next_delivery_dates, key=lambda x: x['date'])
                    next_delivery_date = earliest_delivery['date']
                    best_next_delivery_date = earliest_delivery['best_date']
                    print(f"\nPattern-based next delivery: {next_delivery_date} ({earliest_delivery['group_name']})")
                    
                    # For pattern-based predictions, just write the summary without detailed forecast
                    # (avoid showing 0-unit "deliveries" in the spreadsheet)
                    empty_forecast = pd.DataFrame()
                    
                    self.write_forecast_to_sheets(empty_forecast, next_delivery_date, all_next_delivery_dates, best_next_delivery_date)
                    print("Pattern-based forecasting completed successfully!")
                else:
                    print("No forecasts generated - insufficient data for all groups")
            
        except Exception as e:
            print(f"Error in forecasting pipeline: {str(e)}")
            raise

def main():
    """Main function for script execution"""
    print("Fieldwork Delivery Forecasting")
    print("=" * 50)
    
    # Initialize forecaster (will use environment variable for credentials)
    forecaster = FieldworkForecaster()
    
    # Run the forecasting pipeline
    forecaster.run_forecast()

if __name__ == "__main__":
    main()

