"""Solar Power Generation Forecasting – Streamlit Web App

A day-ahead forecasting tool for solar PV plants.
Upload 7+ days of PEA metering data, and the system predicts
the next 96 fifteen-minute intervals using XGBoost + weather APIs.
"""

import io
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STEPS_PER_DAY = 96
MIN_HISTORY_DAYS = 7
MODEL_PATH = Path("models/xgboost_v2.pkl")

# S01 defaults
S01_LAT = 13.82
S01_LON = 102.16
S01_CAPACITY = 6000


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Solar Power Forecast",
    page_icon="☀️",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# Weather API fetching
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weather_for_date(forecast_date, lat, lon):
    """Fetch weather from Open-Meteo + NASA POWER for any location."""
    import requests

    start = forecast_date - datetime.timedelta(days=7)
    end = forecast_date

    meteo_vars = [
        "shortwave_radiation", "direct_radiation", "diffuse_radiation",
        "direct_normal_irradiance", "sunshine_duration",
        "temperature_2m", "dewpoint_2m", "apparent_temperature",
        "relative_humidity_2m", "vapour_pressure_deficit",
        "pressure_msl", "precipitation",
        "cloudcover", "cloudcover_low", "cloudcover_mid", "cloudcover_high",
        "windspeed_10m", "windgusts_10m",
    ]

    meteo_rename = {
        "shortwave_radiation": "fcst_ghi",
        "direct_radiation": "fcst_direct_rad",
        "diffuse_radiation": "fcst_diffuse_rad",
        "direct_normal_irradiance": "fcst_dni",
        "sunshine_duration": "fcst_sunshine_sec",
        "temperature_2m": "fcst_temp",
        "dewpoint_2m": "fcst_dewpoint",
        "apparent_temperature": "fcst_apparent_temp",
        "relative_humidity_2m": "fcst_humidity",
        "vapour_pressure_deficit": "fcst_vpd",
        "pressure_msl": "fcst_pressure",
        "precipitation": "fcst_precip",
        "cloudcover": "fcst_cloud",
        "cloudcover_low": "fcst_cloud_low",
        "cloudcover_mid": "fcst_cloud_mid",
        "cloudcover_high": "fcst_cloud_high",
        "windspeed_10m": "fcst_wind",
        "windgusts_10m": "fcst_windgusts",
    }

    try:
        resp = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat, "longitude": lon,
                "start_date": str(start), "end_date": str(end),
                "hourly": ",".join(meteo_vars), "timezone": "Asia/Bangkok",
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").rename(columns=meteo_rename)
    except Exception:
        return None

    # NASA POWER (optional)
    try:
        resp = requests.get(
            "https://power.larc.nasa.gov/api/temporal/hourly/point",
            params={
                "parameters": "ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,ALLSKY_KT,T2M,RH2M,WS2M",
                "community": "RE", "longitude": lon, "latitude": lat,
                "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"),
                "format": "JSON", "time-standard": "LST",
            },
            timeout=120,
        )
        resp.raise_for_status()
        nasa_data = resp.json()["properties"]["parameter"]
        records = {}
        for param_name, values in nasa_data.items():
            for ts_key, value in values.items():
                if ts_key not in records:
                    records[ts_key] = {}
                records[ts_key][param_name] = value if value != -999.0 else np.nan
        df_nasa = pd.DataFrame.from_dict(records, orient="index")
        df_nasa.index = pd.to_datetime(df_nasa.index, format="%Y%m%d%H")
        df_nasa = df_nasa.rename(columns={
            "ALLSKY_SFC_SW_DWN": "nasa_ghi", "CLRSKY_SFC_SW_DWN": "nasa_clearsky_ghi",
            "ALLSKY_KT": "nasa_clearness_index", "T2M": "nasa_temp",
            "RH2M": "nasa_humidity", "WS2M": "nasa_wind",
        })
        df = df.join(df_nasa, how="left")
    except Exception:
        pass

    # Resample to 15-minute
    full_idx = pd.date_range(
        df.index.min(),
        df.index.max().normalize() + pd.Timedelta(hours=23, minutes=45),
        freq="15min",
    )
    df = df.reindex(full_idx).interpolate(method="time", limit=4)

    for col in df.columns:
        if any(kw in col for kw in ["ghi", "rad", "dni", "clearsky"]):
            df[col] = df[col].clip(lower=0)
        if "cloud" in col:
            df[col] = df[col].clip(0, 100)
        if "humidity" in col:
            df[col] = df[col].clip(0, 100)

    # Clear-sky index
    try:
        import pvlib
        loc = pvlib.location.Location(lat, lon, tz="Asia/Bangkok", altitude=50)
        cs = loc.get_clearsky(df.index, model="ineichen")
        df["clearsky_ghi"] = cs["ghi"]
        safe_cs = cs["ghi"].replace(0, np.nan)
        df["clear_sky_index"] = (df["fcst_ghi"] / safe_cs).clip(0, 1.5)
    except ImportError:
        pass

    df.index.name = "timestamp"
    return df


# ---------------------------------------------------------------------------
# PEA xlsx parser
# ---------------------------------------------------------------------------
def parse_pea_xlsx(uploaded_file):
    xls = pd.ExcelFile(uploaded_file)
    all_frames = []

    for sheet_name in xls.sheet_names:
        if sheet_name.lower() == "instructions":
            continue
        try:
            df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
            dt_col = None
            for col in df_sheet.columns:
                if "date" in str(col).lower() or "time" in str(col).lower():
                    dt_col = col
                    break
            if dt_col is None:
                dt_col = df_sheet.columns[0]

            df_sheet["datetime"] = pd.to_datetime(df_sheet[dt_col], errors="coerce")
            df_sheet = df_sheet.dropna(subset=["datetime"])

            col_map = {}
            for target_col in ["PVTEMP1", "PVTEMP2", "AMBTEMP", "PYRANO1",
                               "PYRANO2", "WINDSPEED", "P"]:
                for src_col in df_sheet.columns:
                    if str(src_col).strip().upper() == target_col:
                        col_map[src_col] = target_col
                        break

            df_sheet = df_sheet.rename(columns=col_map).set_index("datetime")
            keep = [c for c in ["PVTEMP1", "PVTEMP2", "AMBTEMP", "PYRANO1",
                                "PYRANO2", "WINDSPEED", "P"] if c in df_sheet.columns]
            all_frames.append(df_sheet[keep])
        except Exception:
            continue

    if not all_frames:
        return pd.DataFrame()

    result = pd.concat(all_frames).sort_index()
    result = result[~result.index.duplicated(keep="first")]
    for col in result.columns:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def preprocess_plant_data(df_raw):
    df = df_raw.copy()

    if "PYRANO1" in df.columns and "PYRANO2" in df.columns:
        df["irradiance"] = df[["PYRANO1", "PYRANO2"]].mean(axis=1)
    elif "PYRANO1" in df.columns:
        df["irradiance"] = df["PYRANO1"]

    if "PVTEMP1" in df.columns and "PVTEMP2" in df.columns:
        df["pv_temp"] = df[["PVTEMP1", "PVTEMP2"]].mean(axis=1)
    elif "PVTEMP1" in df.columns:
        df["pv_temp"] = df["PVTEMP1"]

    if "WINDSPEED" in df.columns:
        df.loc[df["WINDSPEED"] >= 60, "WINDSPEED"] = np.nan
        df.loc[df["WINDSPEED"] > 20, "WINDSPEED"] = np.nan
    if "AMBTEMP" in df.columns:
        df.loc[df["AMBTEMP"] >= 50, "AMBTEMP"] = np.nan

    df_15 = df.resample("15min").mean()
    p_count = df["P"].resample("15min").count()
    df_15.loc[p_count < 2, "P"] = np.nan
    return df_15


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_app_features(df_15, forecast_date, forecast_weather=None):
    day_start = pd.Timestamp(forecast_date)
    target_times = pd.date_range(day_start, periods=STEPS_PER_DAY, freq="15min")
    out = pd.DataFrame(index=target_times)

    for days in [1, 2, 7]:
        lag_times = target_times - pd.Timedelta(days=days)
        out[f"P_lag{days}"] = df_15["P"].reindex(lag_times).values

    history = []
    for d in range(1, 8):
        history.append(df_15["P"].reindex(target_times - pd.Timedelta(days=d)).values)
    with np.errstate(all="ignore"):
        out["P_roll7"] = np.nanmean(np.array(history), axis=0)

    lag1_times = target_times - pd.Timedelta(days=1)
    for name, col in [("irr_lag1", "irradiance"), ("pv_temp_lag1", "pv_temp"),
                      ("amb_lag1", "AMBTEMP"), ("wind_lag1", "WINDSPEED")]:
        out[name] = df_15[col].reindex(lag1_times).values if col in df_15.columns else np.nan

    slot = target_times.hour * 4 + target_times.minute // 15
    for prefix, value, period in [("interval", slot, 96),
                                   ("doy", target_times.dayofyear, 365.25),
                                   ("month", target_times.month, 12)]:
        out[f"{prefix}_sin"] = np.sin(2 * np.pi * value / period)
        out[f"{prefix}_cos"] = np.cos(2 * np.pi * value / period)
    out["dow"] = target_times.dayofweek

    daily_energy = (df_15["P"].groupby(df_15.index.normalize()).sum() * 0.25)
    energy_vals = []
    for d in range(1, 8):
        energy_vals.append(daily_energy.get(
            pd.Timestamp(forecast_date - datetime.timedelta(days=d)), np.nan
        ))
    valid_e = [e for e in energy_vals if not (isinstance(e, float) and np.isnan(e))]
    out["energy_roll7_mean"] = np.mean(valid_e) if len(valid_e) >= 3 else np.nan
    out["energy_roll7_std"] = np.std(valid_e) if len(valid_e) >= 3 else np.nan

    if forecast_weather is not None:
        for col in forecast_weather.columns:
            out[col] = forecast_weather[col].reindex(target_times).values

        if "fcst_ghi" in out.columns and "irr_lag1" in out.columns:
            out["fcst_ghi_change"] = out["fcst_ghi"] - out["irr_lag1"]
        if "fcst_diffuse_rad" in out.columns and "fcst_ghi" in out.columns:
            out["fcst_diffuse_ratio"] = (out["fcst_diffuse_rad"] / out["fcst_ghi"].replace(0, np.nan)).clip(0, 1)
        if "fcst_cloud" in out.columns and "fcst_ghi" in out.columns:
            out["fcst_cloud_x_ghi"] = out["fcst_cloud"] * out["fcst_ghi"] / 100
        if "fcst_temp" in out.columns:
            out["fcst_temp_sq"] = out["fcst_temp"] ** 2

        prev_day = forecast_date - datetime.timedelta(days=1)
        prev_mask = df_15.index.normalize() == pd.Timestamp(prev_day)
        if prev_mask.any():
            if "irradiance" in df_15.columns:
                out["irr_variability_lag1"] = df_15.loc[prev_mask, "irradiance"].std()
            out["peak_power_lag1"] = df_15.loc[prev_mask, "P"].max()
            out["hours_above_1mw_lag1"] = (df_15.loc[prev_mask, "P"] > 1000).sum()

    return out


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def run_forecast(model_artifact, features_df, plant_capacity):
    model = model_artifact["model"]
    expected = model_artifact["feature_names"]

    X = features_df.reindex(columns=expected)
    for col in expected:
        if col not in X.columns:
            X[col] = np.nan
    X = X[expected]

    y_pred = np.clip(model.predict(X), 0, plant_capacity)

    result = pd.DataFrame({"timestamp": features_df.index, "power_kw": y_pred})

    if model_artifact.get("model_q10") is not None and model_artifact.get("model_q90") is not None:
        q10 = np.clip(model_artifact["model_q10"].predict(X), 0, plant_capacity)
        q90 = np.clip(model_artifact["model_q90"].predict(X), 0, plant_capacity)
        result["lower_bound"] = np.minimum(q10, q90)
        result["upper_bound"] = np.maximum(q10, q90)
    else:
        result["lower_bound"] = (y_pred * 0.7).clip(0)
        result["upper_bound"] = (y_pred * 1.3).clip(0, plant_capacity)

    result["time"] = result["timestamp"].dt.strftime("%H:%M")
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_forecast(result, forecast_date, plant_capacity):
    fig, ax = plt.subplots(figsize=(10, 4))
    hours = np.arange(len(result)) / 4

    ax.fill_between(hours, result["lower_bound"], result["upper_bound"],
                    alpha=0.15, color="#2196F3", label="80% interval")
    ax.plot(hours, result["power_kw"], color="#FF9800", linewidth=2, label="Forecast")

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Power (kW)")
    ax.set_title(f"Day-Ahead Forecast — {forecast_date}")
    ax.set_xlim(0, 24)
    ax.set_ylim(0, plant_capacity * 1.05)
    ax.set_xticks(range(0, 25, 3))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Template generator
# ---------------------------------------------------------------------------
def generate_template(forecast_date):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for days_back in range(7, 0, -1):
            day = forecast_date - datetime.timedelta(days=days_back)
            timestamps = pd.date_range(f"{day} 00:00", f"{day} 23:55", freq="5min")
            df_t = pd.DataFrame({
                "Date/Time": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
                "PVTEMP1": "", "PVTEMP2": "", "AMBTEMP": "",
                "PYRANO1": "", "PYRANO2": "", "WINDSPEED": "", "P": "",
            })
            df_t.to_excel(writer, sheet_name=str(day.day), index=False)
            ws = writer.sheets[str(day.day)]
            ws.column_dimensions["A"].width = 22

        instructions = pd.DataFrame({
            "Column": ["Date/Time", "PVTEMP1", "PVTEMP2", "AMBTEMP",
                        "PYRANO1", "PYRANO2", "WINDSPEED", "P"],
            "Unit": ["datetime", "°C", "°C", "°C", "W/m²", "W/m²", "m/s", "kW"],
            "Description": [
                "Timestamp at 5-min intervals", "PV module temp sensor 1",
                "PV module temp sensor 2", "Ambient temperature",
                "Irradiance sensor 1", "Irradiance sensor 2",
                "Wind speed", "Power output",
            ],
        })
        instructions.to_excel(writer, sheet_name="Instructions", index=False)
        ws = writer.sheets["Instructions"]
        ws.column_dimensions["C"].width = 30
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    # Header
    st.title("☀️ Solar Power Generation Forecast")
    st.caption("Day-ahead forecasting · 96 intervals at 15-minute resolution")

    # Load model
    artifact = load_model()
    if artifact is None:
        st.error("Model not found. Place `models/xgboost_v2.pkl` in the project root.")
        return

    st.divider()

    # --- Plant selection ---
    plant_option = st.radio(
        "Select plant",
        ["S01 — Sa Kaeo (6 MW)", "Other location"],
        horizontal=True,
    )

    if plant_option == "S01 — Sa Kaeo (6 MW)":
        latitude = S01_LAT
        longitude = S01_LON
        plant_capacity = S01_CAPACITY
    else:
        st.info(
            "The model was trained on S01 (Sa Kaeo) data. "
            "Predictions for other locations are approximate and would improve with site-specific retraining."
        )
        col_lat, col_lon, col_cap = st.columns(3)
        with col_lat:
            latitude = st.number_input("Latitude", value=13.82, format="%.4f")
        with col_lon:
            longitude = st.number_input("Longitude", value=102.16, format="%.4f")
        with col_cap:
            plant_capacity = st.number_input("Capacity (kW)", value=6000, min_value=100, max_value=50000, step=100)

    st.divider()

    # --- Input section ---
    col_date, col_upload = st.columns([1, 2])

    with col_date:
        forecast_date = st.date_input(
            "Forecast date",
            value=datetime.date.today() + datetime.timedelta(days=1),
        )
        template_xlsx = generate_template(forecast_date)
        st.download_button(
            "📋 Download template",
            template_xlsx,
            f"template_{forecast_date}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with col_upload:
        uploaded_file = st.file_uploader(
            f"Upload plant data (7+ days before {forecast_date})",
            type=["xlsx", "xls"],
        )

    if not uploaded_file:
        st.info("Upload a PEA metering file to generate the forecast.")
        return

    # --- Parse ---
    df_parsed = parse_pea_xlsx(uploaded_file)
    if df_parsed.empty:
        st.error("Could not parse the file. Check column names match the template.")
        return

    data_start = df_parsed.index.min().date()
    data_end = df_parsed.index.max().date()
    st.success(f"✅ {len(df_parsed):,} records · {data_start} to {data_end}")

    # Validate
    min_required = forecast_date - datetime.timedelta(days=MIN_HISTORY_DAYS)
    if data_end < min_required:
        st.error(f"Data ends {data_end}, need at least up to {forecast_date - datetime.timedelta(days=1)}.")
        return

    # --- Process and predict ---
    with st.spinner("Processing data and fetching weather..."):
        df_15 = preprocess_plant_data(df_parsed)
        forecast_weather = fetch_weather_for_date(forecast_date, latitude, longitude)
        features = build_app_features(df_15, forecast_date, forecast_weather)
        result = run_forecast(artifact, features, plant_capacity)

    st.divider()

    # --- Results ---
    fig = plot_forecast(result, forecast_date, plant_capacity)
    st.pyplot(fig)
    plt.close()

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Peak power", f"{result['power_kw'].max():.0f} kW")
    c2.metric("Daily energy", f"{result['power_kw'].sum() * 0.25:.0f} kWh")
    c3.metric("Peak hour", result.loc[result["power_kw"].idxmax(), "time"])
    c4.metric("Capacity factor", f"{result['power_kw'].mean() / plant_capacity * 100:.1f}%")

    st.divider()

    # Predictions table
    st.subheader("96-interval predictions")
    display_df = result[["time", "power_kw", "lower_bound", "upper_bound"]].copy()
    display_df.columns = ["Time", "Power (kW)", "Lower (kW)", "Upper (kW)"]
    display_df["Power (kW)"] = display_df["Power (kW)"].round(1)
    display_df["Lower (kW)"] = display_df["Lower (kW)"].round(1)
    display_df["Upper (kW)"] = display_df["Upper (kW)"].round(1)
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)

    st.divider()

    # Download row
    c1, c2 = st.columns(2)

    csv_buf = io.StringIO()
    result[["timestamp", "power_kw", "lower_bound", "upper_bound"]].to_csv(csv_buf, index=False)
    c1.download_button("📄 Download CSV", csv_buf.getvalue(),
                       f"forecast_{forecast_date}.csv", "text/csv")

    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
        result[["timestamp", "time", "power_kw", "lower_bound", "upper_bound"]].to_excel(
            writer, sheet_name="Forecast", index=False)
    c2.download_button("📊 Download Excel", xlsx_buf.getvalue(),
                       f"forecast_{forecast_date}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    main()
