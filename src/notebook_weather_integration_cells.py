"""
Reference: Notebook cells to add for weather API integration
=============================================================

Copy these into your notebook at the indicated positions.
"""

# ===========================================================================
# CELL 1 — Add AFTER cell 4 (setup/config), BEFORE data loading
# Cell type: markdown
# ===========================================================================
"""
## 0.1 Forecast weather configuration

Day-ahead weather forecasts from Open-Meteo Historical Weather API
(ERA5 reanalysis) provide target-day conditions that would be available
to an operator before issuing the forecast. Run `fetch_weather.py` once
to download the data.
"""

# ===========================================================================
# CELL 2 — Add right after the markdown above
# Cell type: code
# ===========================================================================

# Path to the Open-Meteo weather CSV (run fetch_weather.py first)
FORECAST_WEATHER_PATH = "data/forecast_weather_15min.csv"

# ===========================================================================
# CELL 3 — Add AFTER resampling/cleaning (after cell ~49, day eligibility),
#           BEFORE the existing feature engineering section
# Cell type: markdown
# ===========================================================================
"""
## 3.5 Load day-ahead forecast weather

Open-Meteo ERA5 reanalysis data at 15-minute resolution, aligned with
S01 timestamps. These represent the weather conditions the forecast
model would have access to before the target day.
"""

# ===========================================================================
# CELL 4 — Add right after the markdown above
# Cell type: code
# ===========================================================================

from src.data_quality import load_forecast_weather

forecast_weather = load_forecast_weather(FORECAST_WEATHER_PATH)
print(f"\nForecast weather date range: {forecast_weather.index.min()} to {forecast_weather.index.max()}")
print(f"Columns: {forecast_weather.columns.tolist()}")
print(f"\nSample (noon values):")
display(forecast_weather.between_time("11:00", "13:00").head(10))

# Quick correlation check: how well does Open-Meteo GHI match on-site PYRANO?
if 'fcst_ghi' in forecast_weather.columns:
    merged = df_clean_15min[['irradiance', 'P']].join(forecast_weather['fcst_ghi'], how='inner')
    merged = merged.dropna()
    corr_irr = merged['irradiance'].corr(merged['fcst_ghi'])
    corr_p = merged['P'].corr(merged['fcst_ghi'])
    print(f"\nCorrelation: Open-Meteo GHI vs on-site irradiance = {corr_irr:.3f}")
    print(f"Correlation: Open-Meteo GHI vs power output = {corr_p:.3f}")

# ===========================================================================
# CELL 5 — REPLACE the existing feature engineering cell (cell ~57)
#           that calls build_forecast_features
# Cell type: code
# ===========================================================================

from src.data_quality import build_forecast_features
build_features = build_forecast_features

df_full_imputed = pd.concat([df_train_raw, df_test_raw]).sort_index()  # raw, despite legacy name
df_features = build_features(df_full_imputed, forecast_weather=forecast_weather)
df_train_feat = df_features.loc[df_features.index.normalize() <= train_end_date].copy()
df_test_feat = df_features.loc[df_features.index.normalize() >= test_start_date].copy()
target_col = "P"
feature_names = [c for c in df_features.columns if c != target_col]
print(f"Features ({len(feature_names)}): {feature_names}")

# ===========================================================================
# CELL 6 — Add AFTER the model results section (after cell ~67)
# Cell type: markdown
# ===========================================================================
"""
## 8.1 Impact of forecast weather features

Compare model performance with and without day-ahead weather forecasts
to quantify the value of forecast information for XGBoost vs Ridge.
"""

# ===========================================================================
# CELL 7 — Add right after the markdown above
# Cell type: code
# ===========================================================================

# Summary: which features helped?
fcst_cols_in_data = [c for c in feature_names if c.startswith("fcst_") or c == "clear_sky_index"]
print(f"Forecast features used: {fcst_cols_in_data}")
print()

# Filter the results table to show the ablation comparison
if 'results_df' in advanced:
    comparison = advanced['results_df']
    # Show only 24h results for clean comparison
    full_day = comparison[comparison['Model'].str.contains('24h')]
    print("Model comparison (24h):")
    display(full_day[['Model', 'MAE', 'RMSE', 'nRMSE', 'R2', 'Skill']].to_string(index=False))

# ===========================================================================
# CELL 8 — OPTIONAL: Add after the walk-forward section
#           to see monthly forecast weather impact
# Cell type: code
# ===========================================================================

# Walk-forward with models (now includes forecast features automatically)
from src.modeling import walk_forward_with_models, plot_walk_forward_model_mae

wf_models = walk_forward_with_models(
    df_clean_raw, feature_names, 
    lambda raw: build_features(raw, forecast_weather=forecast_weather),
    output_dir=OUTPUT_DIR,
    min_train_months=6,
    impute_feature_columns=feature_names,
    search_n_iter=XGB_SEARCH_ITER,
    capacity_kw=PLANT_CAPACITY_KW,
    random_state=RANDOM_STATE,
)
display(wf_models)
plot_walk_forward_model_mae(wf_models, OUTPUT_DIR)

# Seasonal summary with forecast features
season_summary = wf_models.groupby("Season")[
    ["Ridge MAE", "XGBoost MAE"]
].mean().round(2).reindex(["Hot", "Rainy", "Cool"])
print("\nSeasonal MAE with forecast weather features:")
display(season_summary)


# ===========================================================================
# HOW TO RUN fetch_weather.py BEFORE THE NOTEBOOK
# ===========================================================================
"""
Step 1: Install dependencies (if not already installed)
    pip install requests pvlib

Step 2: Run the fetch script from your project root
    python -m src.fetch_weather

    This creates data/forecast_weather_15min.csv (~15 MB)
    Takes about 30 seconds to download.

Step 3: Open and run the notebook as usual
    The notebook will load the CSV and add forecast features automatically.

If you don't have internet or Open-Meteo is down:
    The notebook still works without forecast weather.
    Just set forecast_weather = None in the feature engineering cell.
"""
