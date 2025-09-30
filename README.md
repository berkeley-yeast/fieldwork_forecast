# Fieldwork Delivery Forecasting

This repository contains an automated forecasting system for Fieldwork delivery predictions using machine learning with skforecast and LightGBM.

## Overview

The system:
- Pulls data from Google Sheets ("large_account_na_data" tab "Fieldwork")
- Trains a forecasting model using LightGBM with complex seasonality patterns
- Predicts the next delivery date within 2 weeks
- Writes predictions back to the same Google Sheet starting at column F (index 6)
- Runs automatically every Monday via GitHub Actions

## Data Structure

The system expects the following columns in the Google Sheet:
- **date**: Date of delivery
- **by_units**: Volume delivered  
- **strain**: Type of strain
- **strain_type**: Type of strain classification

## Output Structure

Predictions are written starting at column F:
- **Column F**: Forecast Date
- **Column G**: Predicted Units
- **Column H**: Generated On
- **Column J**: Next Delivery Date
- **Column K**: Days Until Delivery

## Setup Instructions

### 1. GitHub Repository Setup

1. Set your GitHub token as an environment variable:
   ```bash
   export GITHUB_TOKEN=your_github_token_here
   ```

2. Run the setup script:
   ```bash
   ./setup_github_repo.sh
   ```

**⚠️ SECURITY NOTE**: Never commit GitHub tokens to the repository. Always use environment variables.

### 2. Required GitHub Secrets

Add the following secrets to your GitHub repository (Settings > Secrets and variables > Actions):

#### GOOGLE_SHEETS_CREDENTIALS
The complete Google Cloud Platform service account JSON credentials. This should be the entire JSON file content as a string.

Example format:
```json
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "key-id",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "service-account@project.iam.gserviceaccount.com",
  "client_id": "client-id",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/service-account%40project.iam.gserviceaccount.com"
}
```

### 3. Google Cloud Platform Setup

1. Create a new project in Google Cloud Console
2. Enable the Google Sheets API and Google Drive API
3. Create a service account with the following permissions:
   - Google Sheets API access
   - Google Drive API access (for accessing shared sheets)
4. Download the service account JSON key
5. Share your Google Sheet with the service account email address (found in the JSON)

### 4. Google Sheets Permissions

Ensure the service account email has **Editor** access to the "large_account_na_data" Google Sheet.

## Model Details

### Forecasting Approach
The system uses an **ensemble approach** optimized for intermittent demand:

1. **Croston's Method (SBA variant)** - Specialized statistical method for sparse/intermittent demand patterns
2. **Machine Learning Models**:
   - Gradient Boosting Regressor for delivery probability
   - Gradient Boosting Regressor for delivery size estimation
3. **Pattern-based Analysis** - Historical interval analysis as backup

**Ensemble Weighting**: Croston's (40%) + ML (40%) + Pattern (20%)

### Features
- **Time-based**: day of week, month, quarter, month-end/start indicators
- **Trend**: days since last delivery (critical for intermittent forecasting)
- **Rolling averages**: 7, 14, 28-day windows
- **Categorical**: strain and strain_type encoding

### Forecast Horizon
28 days (4 weeks) with confidence intervals (±15%)

### Model Validation
- Time series cross-validation (80/20 split)
- Metrics tracked: MAE, RMSE, delivery detection accuracy
- Results written to Google Sheets for monitoring

### Model Training
The model automatically:
1. Handles missing dates by filling with zeros
2. Creates daily aggregations from the raw data
3. Generates time-based features for seasonality
4. Detects intermittency patterns and applies appropriate methods
5. Validates using time series cross-validation
6. Provides confidence scores and prediction intervals

## Automation Schedule

- **Frequency**: Every Monday at 8:00 AM UTC
- **Trigger**: GitHub Actions cron schedule
- **Manual Trigger**: Available via GitHub Actions UI

## Testing

### Local Testing (Not Recommended)
Local testing is discouraged as per requirements. All testing should be done via GitHub Actions.

### GitHub Actions Testing
1. Use the "workflow_dispatch" trigger to manually run the workflow
2. Check the Actions tab for execution logs
3. Verify results in the Google Sheet

## Troubleshooting

### Common Issues

1. **Google Sheets Access Denied**
   - Verify service account has access to the sheet
   - Check that the sheet name and tab name are correct
   - Ensure Google Sheets and Drive APIs are enabled

2. **Missing Dependencies**
   - Check requirements.txt for all necessary packages
   - Verify Python version compatibility (3.11)

3. **Forecast Errors**
   - Check data quality in the source sheet
   - Ensure date formats are consistent
   - Verify numeric data in by_units column

### Logs and Debugging

- GitHub Actions logs are available in the Actions tab
- Failed runs will upload log artifacts for debugging
- Check the Google Sheet for error messages in forecast columns

## Files Structure

```
fieldwork_forecast/
├── forecast_fieldwork.py           # Main forecasting script
├── requirements.txt                # Python dependencies
├── .github/
│   └── workflows/
│       └── forecast.yml           # GitHub Actions workflow
└── README.md                      # This file
```

## Contact

For issues or questions, contact the berkeley-yeast team or check the GitHub Actions logs for detailed error information.

