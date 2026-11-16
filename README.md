# ☀️ Solar Power Generation Forecasting

Day-ahead solar power forecasting system for PV plants in Thailand. Predicts 96 consecutive 15-minute power outputs using XGBoost with historical generation data, on-site meteorological observations, and day-ahead weather forecasts.

## Overview

- **Model:** XGBoost with Optuna Bayesian hyperparameter tuning (100 trials)
- **Features:** 50 input features — historical power lags, on-site weather, Open-Meteo forecast, NASA POWER satellite data, calendar encodings, and derived features
- **Baseline:** Ridge Regression (L2-regularized Linear Regression)
- **Data:** 2 years of 5-minute PEA metering data (2023–2024), aggregated to 15-minute intervals
- **Performance:** XGBoost MAE 206 kW, R² 0.925 on chronological hold-out test set

## Web App

A Streamlit web application allows users to upload plant metering data and receive a 96-interval day-ahead forecast with prediction intervals.

**Live:** [solar-forecast.calmriver-877801bc.eastasia.azurecontainerapps.io](https://solar-forecast.calmriver-877801bc.eastasia.azurecontainerapps.io/)

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
├── app.py                  # Streamlit web app
├── Dockerfile              # Container build
├── requirements.txt        # Python dependencies
├── src/
│   ├── data_quality.py     # Data preprocessing and feature engineering
│   ├── modeling.py         # Model training, evaluation, Optuna tuning
│   └── fetch_weather.py    # Open-Meteo and NASA POWER API client
├── models/
│   └── xgboost_v2.pkl      # Trained model artifact
├── notebooks/
│   └── S01_Solar_DayAhead_Forecasting_Progress.ipynb
└── infra/
    └── setup_azure.sh      # Azure Container Apps setup script
```

## Deployment

Deployed on Azure Container Apps with GitHub Actions CI/CD. Pushes to `main` trigger automatic build, test, and deployment.

## Data Sources

| Source | Type | Variables |
|---|---|---|
| PEA (Provincial Electricity Authority) | On-site metering | Power output, irradiance, temperature, wind speed |
| Open-Meteo Archive API | Weather reanalysis | GHI, cloud layers, humidity, pressure, precipitation (18 variables) |
| NASA POWER API | Satellite-derived | GHI, clear-sky GHI, clearness index (6 variables) |

## Team

| Name | Student ID |
|---|---|
| Puritaporn Thongsuvan | st127397 |
| Swel Myat Sandi Min Thein | st127401 |

**Course:** AST02.10 Computer Programming for Data Science and AI  
**Instructor:** Dr. Chantri Polprasert  
**Institution:** Asian Institute of Technology
