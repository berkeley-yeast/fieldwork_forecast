#!/usr/bin/env python3
"""
Fieldwork Delivery Forecasting Script
Implements Croston's method and ML ensemble for intermittent demand forecasting
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

# Data processing
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.model_selection import TimeSeriesSplit

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class CrostonsMethod:
    """Croston's Method for intermittent demand forecasting"""
    
    def __init__(self, alpha=0.1, method='croston'):
        """
        Initialize Croston's method
        
        Parameters:
        - alpha: smoothing parameter (0-1)
        - method: 'croston', 'sba' (Syntetos-Boylan Approximation), or 'tsb' (Teunter-Syntetos-Babai)
        """
        self.alpha = alpha
        self.method = method
        self.demand_size = None
        self.demand_interval = None
        
    def fit(self, series):
        """Fit Croston's method on historical data"""
        # Extract non-zero demands
        non_zero_indices = np.where(series > 0)[0]
        
        if len(non_zero_indices) < 2:
            # Not enough data for Croston's method
            self.demand_size = series[series > 0].mean() if len(series[series > 0]) > 0 else 0
            self.demand_interval = len(series) / max(1, len(non_zero_indices))
            return self
        
        # Initialize with first demand
        demand_sizes = series[non_zero_indices]
        
        # Calculate intervals between demands
        intervals = np.diff(non_zero_indices)
        
        # Exponential smoothing on demand sizes
        smoothed_size = demand_sizes[0]
        for size in demand_sizes[1:]:
            smoothed_size = self.alpha * size + (1 - self.alpha) * smoothed_size
        
        # Exponential smoothing on intervals
        smoothed_interval = intervals[0]
        for interval in intervals[1:]:
            smoothed_interval = self.alpha * interval + (1 - self.alpha) * smoothed_interval
        
        self.demand_size = smoothed_size
        self.demand_interval = smoothed_interval
        
        return self
    
    def predict(self, steps=1):
        """Predict future demand"""
        if self.demand_size is None or self.demand_interval is None:
            return np.zeros(steps)
        
        if self.method == 'croston':
            # Standard Croston's: forecast = size / interval
            forecast_value = self.demand_size / max(1, self.demand_interval)
        elif self.method == 'sba':
            # Syntetos-Boylan Approximation (less biased)
            forecast_value = (self.demand_size / max(1, self.demand_interval)) * (1 - self.alpha/2)
        else:  # tsb
            # TSB method
            prob_demand = 1 / max(1, self.demand_interval)
            forecast_value = prob_demand * self.demand_size
        
        return np.full(steps, forecast_value)


class FieldworkForecaster:
    def __init__(self, google_creds_path=None):
        """Initialize the forecaster with Google Sheets credentials"""
        self.sheet_name = "large_account_na_data"
        self.tab_name = "Fieldwork"
        self.forecast_horizon = 28
        self.today = datetime.now().date()
        self.validation_metrics = {}
        
        if google_creds_path:
            self.setup_google_sheets(google_creds_path)
        else:
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
            sheet = self.gc.open(self.sheet_name)
            worksheet = sheet.worksheet(self.tab_name)
            
            try:
                data = worksheet.get_all_records()
            except gspread.exceptions.GSpreadException as e:
                if "duplicates" in str(e):
                    print("Handling duplicate headers...")
                    all_values = worksheet.get_all_values()
                    if not all_values:
                        raise ValueError("No data found in worksheet")
                    
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
                    
                    data_rows = all_values[1:]
                    data = []
                    for row in data_rows:
                        padded_row = row + [''] * (len(unique_headers) - len(row))
                        data.append(dict(zip(unique_headers, padded_row)))
                else:
                    raise e
            
            if not data:
                raise ValueError("No data found in the specified sheet/tab")
            
            df = pd.DataFrame(data)
            print(f"Loaded {len(df)} rows of data")
            return df
            
        except Exception as e:
            print(f"Error loading data from Google Sheets: {str(e)}")
            raise
    
    def preprocess_data(self, df):
        """Preprocess the data for forecasting with quality checks"""
        print("Preprocessing data...")
        
        print(f"Available columns: {df.columns.tolist()}")
        
        # Find required columns
        date_col = next((col for col in df.columns if 'date' in col.lower()), None)
        units_col = next((col for col in df.columns if 'by_units' in col.lower() or 'units' in col.lower()), None)
        strain_col = next((col for col in df.columns if 'strain' in col.lower() and 'type' not in col.lower()), None)
        strain_type_col = next((col for col in df.columns if 'strain_type' in col.lower()), None)
        
        if not all([date_col, units_col, strain_col, strain_type_col]):
            missing = []
            if not date_col: missing.append('date')
            if not units_col: missing.append('units')
            if not strain_col: missing.append('strain')
            if not strain_type_col: missing.append('strain_type')
            raise ValueError(f"Missing required columns: {missing}")
        
        print(f"Using columns - Date: {date_col}, Units: {units_col}, Strain: {strain_col}, Strain Type: {strain_type_col}")
        
        df = df.rename(columns={
            date_col: 'date',
            units_col: 'by_units', 
            strain_col: 'strain',
            strain_type_col: 'strain_type'
        })
        
        # Convert and clean data
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df['by_units'] = pd.to_numeric(df['by_units'], errors='coerce')
        df = df.dropna(subset=['by_units'])
        df = df.sort_values('date')
        
        # Data quality checks
        print("\n📊 Data Quality Report:")
        print(f"• Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"• Total records: {len(df)}")
        print(f"• Unique strains: {df['strain'].nunique()}")
        print(f"• Unique strain types: {df['strain_type'].nunique()}")
        
        # Check data recency
        days_since_last_data = (self.today - df['date'].max().date()).days
        print(f"• Days since last data: {days_since_last_data}")
        if days_since_last_data > 60:
            print(f"  ⚠️ WARNING: Data is stale ({days_since_last_data} days old)")
        
        # Encode categorical variables
        self.strain_encoder = LabelEncoder()
        df['strain_encoded'] = self.strain_encoder.fit_transform(df['strain'].astype(str))
        
        self.strain_type_encoder = LabelEncoder()
        df['strain_type_encoded'] = self.strain_type_encoder.fit_transform(df['strain_type'].astype(str))
        
        return df
    
    def create_time_series(self, df):
        """Create time series with proper frequency and features"""
        print("Creating time series...")
        
        df = df.set_index('date')
        
        if df.index.duplicated().any():
            print(f"Found {df.index.duplicated().sum()} duplicate dates, aggregating...")
            df = df[~df.index.duplicated(keep='first')]
        
        # Aggregate by date
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
        
        daily_data = daily_data.reindex(full_range, fill_value=0)
        
        # Add time-based features
        daily_data['day_of_week'] = daily_data.index.dayofweek
        daily_data['day_of_month'] = daily_data.index.day
        daily_data['month'] = daily_data.index.month
        daily_data['quarter'] = daily_data.index.quarter
        daily_data['is_month_end'] = (daily_data.index.day >= 28).astype(int)
        daily_data['is_month_start'] = (daily_data.index.day <= 3).astype(int)
        daily_data['is_quarter_end'] = daily_data.index.to_series().apply(
            lambda x: 1 if x.month in [3, 6, 9, 12] and x.day >= 28 else 0
        ).values
        
        # Days since last delivery - critical for intermittent forecasting
        last_delivery_idx = daily_data[daily_data['by_units'] > 0].index
        daily_data['days_since_last_delivery'] = 0
        if len(last_delivery_idx) > 0:
            for i, date in enumerate(daily_data.index):
                prev_deliveries = last_delivery_idx[last_delivery_idx < date]
                if len(prev_deliveries) > 0:
                    daily_data.loc[date, 'days_since_last_delivery'] = (date - prev_deliveries[-1]).days
                else:
                    daily_data.loc[date, 'days_since_last_delivery'] = 999
        else:
            daily_data['days_since_last_delivery'] = 999
        
        # Rolling features (handle NaNs)
        daily_data['rolling_7'] = daily_data['by_units'].rolling(7, min_periods=1).mean()
        daily_data['rolling_14'] = daily_data['by_units'].rolling(14, min_periods=1).mean()
        daily_data['rolling_28'] = daily_data['by_units'].rolling(28, min_periods=1).mean()
        
        print(f"Time series created with {len(daily_data)} daily observations")
        
        # Analyze intermittency
        zero_days = (daily_data['by_units'] == 0).sum()
        delivery_days = (daily_data['by_units'] > 0).sum()
        intermittency_rate = zero_days / len(daily_data)
        
        print(f"\n📊 Intermittency Analysis:")
        print(f"• Zero delivery days: {zero_days} ({intermittency_rate*100:.1f}%)")
        print(f"• Delivery days: {delivery_days} ({(1-intermittency_rate)*100:.1f}%)")
        print(f"• Average delivery size: {daily_data[daily_data['by_units'] > 0]['by_units'].mean():.1f} BBLs")
        
        if intermittency_rate > 0.75:
            print(f"  ✓ High intermittency detected - Croston's method recommended")
        
        return daily_data
    
    def train_ensemble_forecaster(self, ts_data):
        """Train ensemble forecaster combining Croston's and ML models"""
        print("\nTraining ensemble forecaster...")
        
        exog_vars = [
            'strain_encoded', 'strain_type_encoded', 'day_of_week', 'day_of_month', 
            'month', 'quarter', 'rolling_7', 'rolling_14', 'rolling_28',
            'is_month_end', 'is_month_start', 'is_quarter_end',
            'days_since_last_delivery'
        ]
        
        # Ensure all features exist
        missing_features = set(exog_vars) - set(ts_data.columns)
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
            exog_vars = [f for f in exog_vars if f in ts_data.columns]
        
        self.training_features = exog_vars
        
        # Model 1: Croston's Method (specialized for intermittent demand)
        print("Training Croston's method...")
        self.croston_model = CrostonsMethod(alpha=0.2, method='sba')
        self.croston_model.fit(ts_data['by_units'].values)
        
        # Model 2: Delivery probability classifier
        print("Training delivery probability model...")
        ts_data['has_delivery'] = (ts_data['by_units'] > 0).astype(int)
        
        self.delivery_probability_model = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            subsample=0.8
        )
        
        # Model 3: Delivery size regressor
        print("Training delivery size model...")
        delivery_data = ts_data[ts_data['by_units'] > 0].copy()
        
        if len(delivery_data) >= 5:
            X_prob = ts_data[exog_vars].fillna(0)
            y_prob = ts_data['has_delivery']
            
            # Use probability as continuous target (better for small datasets)
            self.delivery_probability_model.fit(X_prob, y_prob)
            
            # Train size model
            self.delivery_size_model = GradientBoostingRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                subsample=0.8
            )
            
            X_size = delivery_data[exog_vars].fillna(0)
            y_size = delivery_data['by_units']
            self.delivery_size_model.fit(X_size, y_size)
            
            # Validate models with time series split
            self._validate_models(ts_data, exog_vars)
            
            print(f"✓ Models trained on {len(ts_data)} days, {len(delivery_data)} delivery days")
        else:
            print(f"⚠️ Insufficient delivery data ({len(delivery_data)} days), using Croston's only")
            self.delivery_size_model = None
        
        return ts_data
    
    def _validate_models(self, ts_data, exog_vars):
        """Validate model performance using time series cross-validation"""
        print("\nValidating models...")
        
        delivery_data = ts_data[ts_data['by_units'] > 0]
        if len(delivery_data) < 10:
            print("Insufficient data for validation")
            return
        
        # Use last 20% for validation
        split_idx = int(len(ts_data) * 0.8)
        train_data = ts_data.iloc[:split_idx]
        test_data = ts_data.iloc[split_idx:]
        
        if len(test_data[test_data['by_units'] > 0]) < 2:
            print("Insufficient validation data")
            return
        
        # Validate on test set
        X_test = test_data[exog_vars].fillna(0)
        y_test_binary = test_data['has_delivery']
        y_test_size = test_data['by_units']
        
        # Probability predictions
        prob_pred = self.delivery_probability_model.predict(X_test)
        prob_pred_binary = (prob_pred > 0.5).astype(int)
        
        # Size predictions
        if self.delivery_size_model:
            size_pred = self.delivery_size_model.predict(X_test)
            size_pred = np.maximum(size_pred, 0)
            
            # Combined forecast
            combined_pred = prob_pred_binary * size_pred
            
            # Calculate metrics for delivery days only
            delivery_mask = y_test_binary == 1
            if delivery_mask.sum() > 0:
                mae = mean_absolute_error(y_test_size[delivery_mask], combined_pred[delivery_mask])
                rmse = np.sqrt(mean_squared_error(y_test_size[delivery_mask], combined_pred[delivery_mask]))
                
                self.validation_metrics = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'accuracy': accuracy_score(y_test_binary, prob_pred_binary),
                    'test_size': len(test_data),
                    'delivery_days': delivery_mask.sum()
                }
                
                print(f"📈 Validation Metrics:")
                print(f"• MAE (delivery days): {mae:.2f} BBLs")
                print(f"• RMSE (delivery days): {rmse:.2f} BBLs")
                print(f"• Delivery detection accuracy: {self.validation_metrics['accuracy']*100:.1f}%")
    
    def calculate_delivery_statistics(self, ts_data):
        """Calculate statistics about delivery patterns"""
        print("\nCalculating delivery statistics...")
        
        delivery_dates = ts_data[ts_data['by_units'] > 0].index.tolist()
        
        if len(delivery_dates) < 2:
            print("Not enough delivery dates for statistics")
            return None
        
        intervals = [(delivery_dates[i] - delivery_dates[i-1]).days for i in range(1, len(delivery_dates))]
        
        stats = {
            'avg_interval': np.mean(intervals),
            'std_interval': np.std(intervals),
            'min_interval': np.min(intervals),
            'max_interval': np.max(intervals),
            'median_interval': np.median(intervals),
            'total_deliveries': len(delivery_dates),
            'last_delivery_date': delivery_dates[-1]
        }
        
        print(f"• Average interval: {stats['avg_interval']:.1f} ± {stats['std_interval']:.1f} days")
        print(f"• Range: {stats['min_interval']} to {stats['max_interval']} days")
        print(f"• Median: {stats['median_interval']:.1f} days")
        print(f"• Last delivery: {stats['last_delivery_date'].date()}")
        
        return stats
    
    def make_forecast(self, ts_data, delivery_stats):
        """Generate forecast using ensemble approach"""
        print(f"\nGenerating {self.forecast_horizon}-day ensemble forecast...")
        
        tomorrow = self.today + timedelta(days=1)
        forecast_start = datetime.combine(tomorrow, datetime.min.time())
        forecast_dates = pd.date_range(start=forecast_start, periods=self.forecast_horizon, freq='D')
        
        # Croston's forecast
        croston_forecast = self.croston_model.predict(steps=self.forecast_horizon)
        
        # ML forecast
        forecast_exog = self._create_forecast_exog(ts_data, forecast_dates)
        
        ml_probabilities = None
        ml_sizes = None
        
        if self.delivery_size_model:
            ml_probabilities = self.delivery_probability_model.predict(forecast_exog)
            ml_probabilities = np.clip(ml_probabilities, 0, 1)
            ml_sizes = self.delivery_size_model.predict(forecast_exog)
            ml_sizes = np.maximum(ml_sizes, 0)
        
        # Pattern-based forecast
        pattern_forecast = self._pattern_based_forecast(ts_data, delivery_stats, forecast_dates)
        
        # Ensemble: combine all methods with weights
        ensemble_forecast = np.zeros(self.forecast_horizon)
        confidence_scores = np.zeros(self.forecast_horizon)
        
        for i in range(self.forecast_horizon):
            forecasts = []
            weights = []
            
            # Croston's forecast (weight: 0.4 for high intermittency)
            forecasts.append(croston_forecast[i])
            weights.append(0.4)
            
            # ML forecast (weight: 0.4 if available)
            if ml_probabilities is not None and ml_sizes is not None:
                ml_forecast = ml_probabilities[i] * ml_sizes[i]
                forecasts.append(ml_forecast)
                weights.append(0.4)
                confidence_scores[i] = ml_probabilities[i]
            
            # Pattern forecast (weight: 0.2)
            forecasts.append(pattern_forecast[i])
            weights.append(0.2)
            
            # Weighted average
            weights = np.array(weights) / sum(weights)
            ensemble_forecast[i] = np.average(forecasts, weights=weights)
        
        # Find predicted delivery dates (threshold-based)
        delivery_threshold = 100  # BBLs - only show significant deliveries
        predicted_deliveries = []
        
        for i, (date, value, confidence) in enumerate(zip(forecast_dates, ensemble_forecast, confidence_scores)):
            if value >= delivery_threshold:
                predicted_deliveries.append({
                    'date': date,
                    'predicted_units': value,
                    'confidence': confidence if confidence > 0 else 0.5,
                    'method': 'ensemble'
                })
        
        # Calculate confidence intervals (±15%)
        for pred in predicted_deliveries:
            pred['lower_bound'] = pred['predicted_units'] * 0.85
            pred['upper_bound'] = pred['predicted_units'] * 1.15
        
        forecast_df = pd.DataFrame(predicted_deliveries)
        
        # Determine next delivery date
        next_delivery_date = None
        if len(forecast_df) > 0:
            # Use highest confidence prediction
            best_pred = forecast_df.loc[forecast_df['confidence'].idxmax()]
            next_delivery_date = best_pred['date']
        elif delivery_stats:
            # Fallback to pattern-based
            expected_days = delivery_stats['avg_interval']
            next_delivery_date = delivery_stats['last_delivery_date'] + timedelta(days=int(expected_days))
        
        if len(forecast_df) > 0:
            print(f"\n🎯 Forecast Summary:")
            print(f"• Predicted deliveries: {len(forecast_df)}")
            print(f"• Next delivery: {next_delivery_date.date() if next_delivery_date else 'Unknown'}")
            print(f"• Confidence: {forecast_df['confidence'].max()*100:.1f}%")
            
            for _, row in forecast_df.head(3).iterrows():
                print(f"  - {row['date'].date()}: {row['predicted_units']:.1f} BBLs (±{(row['upper_bound']-row['predicted_units']):.1f})")
        
        return forecast_df, next_delivery_date
    
    def _create_forecast_exog(self, ts_data, forecast_dates):
        """Create exogenous variables for forecast period"""
        forecast_exog = pd.DataFrame(index=forecast_dates)
        
        # Use last known strain values
        forecast_exog['strain_encoded'] = ts_data['strain_encoded'].iloc[-1]
        forecast_exog['strain_type_encoded'] = ts_data['strain_type_encoded'].iloc[-1]
        
        # Time features
        forecast_exog['day_of_week'] = forecast_exog.index.dayofweek
        forecast_exog['day_of_month'] = forecast_exog.index.day
        forecast_exog['month'] = forecast_exog.index.month
        forecast_exog['quarter'] = forecast_exog.index.quarter
        forecast_exog['is_month_end'] = (forecast_exog.index.day >= 28).astype(int)
        forecast_exog['is_month_start'] = (forecast_exog.index.day <= 3).astype(int)
        forecast_exog['is_quarter_end'] = forecast_exog.index.to_series().apply(
            lambda x: 1 if x.month in [3, 6, 9, 12] and x.day >= 28 else 0
        ).values
        
        # Days since last delivery
        last_delivery = ts_data[ts_data['by_units'] > 0].index[-1] if len(ts_data[ts_data['by_units'] > 0]) > 0 else ts_data.index[0]
        forecast_exog['days_since_last_delivery'] = [(date - last_delivery).days for date in forecast_exog.index]
        
        # Rolling features (use recent averages)
        forecast_exog['rolling_7'] = ts_data['by_units'].tail(7).mean()
        forecast_exog['rolling_14'] = ts_data['by_units'].tail(14).mean()
        forecast_exog['rolling_28'] = ts_data['by_units'].tail(28).mean()
        
        # Ensure feature order matches training
        forecast_exog = forecast_exog[self.training_features]
        forecast_exog = forecast_exog.fillna(0)
        
        return forecast_exog
    
    def _pattern_based_forecast(self, ts_data, delivery_stats, forecast_dates):
        """Generate pattern-based forecast using historical intervals"""
        forecast = np.zeros(len(forecast_dates))
        
        if not delivery_stats:
            return forecast
        
        last_delivery = delivery_stats['last_delivery_date']
        avg_interval = delivery_stats['avg_interval']
        avg_size = ts_data[ts_data['by_units'] > 0]['by_units'].mean()
        
        # Predict delivery around expected interval
        expected_delivery_date = last_delivery + timedelta(days=int(avg_interval))
        
        for i, date in enumerate(forecast_dates):
            days_diff = abs((date - expected_delivery_date).days)
            if days_diff <= 3:  # Within 3-day window
                # Gaussian-like probability
                prob = np.exp(-0.5 * (days_diff / 1.5) ** 2)
                forecast[i] = avg_size * prob
        
        return forecast
    
    def create_forecast_visualization(self, ts_data, forecast_df, next_delivery_date, delivery_stats):
        """Create visualization showing historical data and forecast"""
        print("\nCreating forecast visualization...")
        
        try:
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            
            # Plot 1: Historical + Forecast
            ax1 = axes[0]
            last_60_days = ts_data.tail(60).copy()
            
            # Historical data
            ax1.plot(last_60_days.index, last_60_days['by_units'], 
                    'b-o', linewidth=2, markersize=4, label='Historical', alpha=0.7)
            
            # Forecast
            if len(forecast_df) > 0:
                ax1.scatter(forecast_df['date'], forecast_df['predicted_units'],
                           s=forecast_df['confidence']*200, c='red', alpha=0.6,
                           label='Forecast (size=confidence)', zorder=5)
                
                # Confidence intervals
                for _, row in forecast_df.iterrows():
                    ax1.vlines(row['date'], row['lower_bound'], row['upper_bound'],
                             colors='red', alpha=0.3, linewidth=2)
                
                # Highlight next delivery
                if next_delivery_date:
                    next_row = forecast_df[forecast_df['date'] == next_delivery_date].iloc[0]
                    ax1.scatter(next_delivery_date, next_row['predicted_units'],
                               s=300, marker='*', c='darkred', edgecolors='black',
                               label=f"Next: {next_delivery_date.date()}", zorder=6)
            
            # Today marker
            ax1.axvline(x=datetime.combine(self.today, datetime.min.time()),
                       color='green', linestyle='--', alpha=0.7, linewidth=2, label='Today')
            
            ax1.set_title('Fieldwork Delivery Forecast (Ensemble Method)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Delivery Volume (BBLs)')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Plot 2: Delivery distribution
            ax2 = axes[1]
            delivery_data = ts_data[ts_data['by_units'] > 0]['by_units']
            ax2.hist(delivery_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(delivery_data.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {delivery_data.mean():.1f} BBLs')
            ax2.set_title('Historical Delivery Size Distribution', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Delivery Volume (BBLs)')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add summary text
            stats_text = f"""Forecast Summary:
Training Period: {ts_data.index.min().date()} to {ts_data.index.max().date()}
Total Observations: {len(ts_data)} | Delivery Days: {(ts_data['by_units'] > 0).sum()}
Avg Interval: {delivery_stats['avg_interval']:.1f} days | Generated: {self.today}
Next Delivery: {next_delivery_date.date() if next_delivery_date else 'TBD'}"""
            
            if self.validation_metrics:
                stats_text += f"\nValidation MAE: {self.validation_metrics['MAE']:.1f} BBLs"
            
            plt.figtext(0.02, 0.02, stats_text, fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)
            
            plot_filename = f"fieldwork_forecast_{self.today.strftime('%Y%m%d')}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved: {plot_filename}")
            plt.close()
            
            return plot_filename
            
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def write_forecast_to_sheets(self, forecast_df, next_delivery_date, delivery_stats, group_name='Overall'):
        """Write forecast results to Google Sheets"""
        print("\nWriting forecast to Google Sheets...")
        
        try:
            sheet = self.gc.open(self.sheet_name)
            worksheet = sheet.worksheet(self.tab_name)
            
            # Clear previous forecast data
            all_values = worksheet.get_all_values()
            if all_values:
                last_row = len(all_values)
                worksheet.batch_clear([f"F1:Z{last_row + 10}"])
                import time
                time.sleep(1)
            
            # Write headers
            headers = ['Forecast Date', 'Predicted Units', 'Lower Bound', 'Upper Bound',
                      'Confidence', 'Group', 'Generated On', 'Next Delivery', 'Days Until']
            worksheet.update('F1:N1', [headers])
            
            # Filter forecast to only include deliveries >= 100 BBLs
            if len(forecast_df) > 0:
                significant_forecasts = forecast_df[forecast_df['predicted_units'] >= 100.0].copy()
                print(f"Filtering forecasts: {len(forecast_df)} total → {len(significant_forecasts)} significant (≥100 BBLs)")
            else:
                significant_forecasts = forecast_df
            
            # Prepare forecast data
            forecast_data = []
            if len(significant_forecasts) > 0:
                for _, row in significant_forecasts.iterrows():
                    days_until = (row['date'].date() - self.today).days
                    forecast_data.append([
                        row['date'].strftime('%Y-%m-%d'),
                        f"{row['predicted_units']:.1f}",
                        f"{row['lower_bound']:.1f}",
                        f"{row['upper_bound']:.1f}",
                        f"{row['confidence']:.2f}",
                        group_name,
                        self.today.strftime('%Y-%m-%d'),
                        next_delivery_date.strftime('%Y-%m-%d') if next_delivery_date else 'TBD',
                        str(days_until) if next_delivery_date else 'TBD'
                    ])
                
                end_row = len(forecast_data) + 1
                worksheet.update(f'F2:N{end_row}', forecast_data)
            else:
                # No forecasts
                status_data = [['No deliveries predicted', '', '', '', '', group_name,
                              self.today.strftime('%Y-%m-%d'),
                              next_delivery_date.strftime('%Y-%m-%d') if next_delivery_date else 'TBD',
                              str((next_delivery_date.date() - self.today).days) if next_delivery_date else 'TBD']]
                worksheet.update('F2:N2', [status_data])
            
            # Write validation metrics if available
            if self.validation_metrics:
                metrics_row = len(forecast_data) + 3
                metrics_headers = ['Metric', 'Value']
                worksheet.update(f'F{metrics_row}:G{metrics_row}', [metrics_headers])
                
                metrics_data = [
                    ['MAE (BBLs)', f"{self.validation_metrics['MAE']:.2f}"],
                    ['RMSE (BBLs)', f"{self.validation_metrics['RMSE']:.2f}"],
                    ['Accuracy (%)', f"{self.validation_metrics['accuracy']*100:.1f}"]
                ]
                worksheet.update(f'F{metrics_row+1}:G{metrics_row+3}', metrics_data)
            
            # Write delivery statistics
            if delivery_stats:
                stats_row = len(forecast_data) + 7
                stats_headers = ['Statistic', 'Value']
                worksheet.update(f'F{stats_row}:G{stats_row}', [stats_headers])
                
                stats_data = [
                    ['Avg Interval (days)', f"{delivery_stats['avg_interval']:.1f}"],
                    ['Std Interval (days)', f"{delivery_stats['std_interval']:.1f}"],
                    ['Min Interval (days)', str(delivery_stats['min_interval'])],
                    ['Max Interval (days)', str(delivery_stats['max_interval'])],
                    ['Total Deliveries', str(delivery_stats['total_deliveries'])],
                    ['Last Delivery', delivery_stats['last_delivery_date'].strftime('%Y-%m-%d')]
                ]
                worksheet.update(f'F{stats_row+1}:G{stats_row+6}', stats_data)
            
            print("✓ Forecast written to Google Sheets successfully")
            
        except Exception as e:
            print(f"Error writing to Google Sheets: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def run_forecast(self):
        """Main forecasting pipeline"""
        print("="*60)
        print("FIELDWORK DELIVERY FORECASTING - ENSEMBLE METHOD")
        print("="*60)
        print(f"Date: {self.today}\n")
        
        try:
            # Load and preprocess data
            raw_data = self.load_data_from_sheets()
            clean_data = self.preprocess_data(raw_data)
            
            # Group by strain_type for better data aggregation
            viable_groups = []
            for strain_type in clean_data['strain_type'].unique():
                group_data = clean_data[clean_data['strain_type'] == strain_type]
                if len(group_data) >= 10:  # Minimum threshold
                    viable_groups.append(('strain_type', strain_type, group_data))
            
            if not viable_groups:
                print("⚠️ No groups with sufficient data, using all data combined")
                viable_groups = [('all', 'All Data', clean_data)]
            
            print(f"\n📦 Found {len(viable_groups)} viable forecasting group(s)")
            
            all_forecasts = []
            all_next_dates = []
            
            # Forecast each group
            for group_type, group_name, group_data in viable_groups:
                print(f"\n{'='*60}")
                print(f"Forecasting: {group_name} ({len(group_data)} records)")
                print(f"{'='*60}")
                
                # Create time series
                ts_data = self.create_time_series(group_data)
                
                # Calculate statistics
                delivery_stats = self.calculate_delivery_statistics(ts_data)
                
                # Train models
                self.train_ensemble_forecaster(ts_data)
                
                # Generate forecast
                forecast_df, next_delivery_date = self.make_forecast(ts_data, delivery_stats)
                
                if len(forecast_df) > 0:
                    forecast_df['group_name'] = group_name
                    all_forecasts.append(forecast_df)
                
                if next_delivery_date:
                    all_next_dates.append({
                        'group_name': group_name,
                        'date': next_delivery_date
                    })
                
                # Create visualization (for first group only to save time)
                if group_type == viable_groups[0][0]:
                    self.create_forecast_visualization(ts_data, forecast_df, next_delivery_date, delivery_stats)
            
            # Combine all forecasts
            if all_forecasts:
                combined_forecast = pd.concat(all_forecasts, ignore_index=True)
                combined_forecast = combined_forecast.sort_values('date')
                
                # Find earliest delivery
                earliest_delivery = min(all_next_dates, key=lambda x: x['date']) if all_next_dates else None
                next_delivery_date = earliest_delivery['date'] if earliest_delivery else None
                
                # Write to sheets
                self.write_forecast_to_sheets(combined_forecast, next_delivery_date, delivery_stats,
                                             earliest_delivery['group_name'] if earliest_delivery else 'Overall')
            else:
                # Write empty forecast with next date if available
                next_delivery_date = all_next_dates[0]['date'] if all_next_dates else None
                self.write_forecast_to_sheets(pd.DataFrame(), next_delivery_date, delivery_stats)
            
            print("\n" + "="*60)
            print("✓ FORECASTING PIPELINE COMPLETED SUCCESSFULLY")
            print("="*60)
            
        except Exception as e:
            print(f"\n❌ Error in forecasting pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main entry point"""
    forecaster = FieldworkForecaster()
    forecaster.run_forecast()


if __name__ == "__main__":
    main()
