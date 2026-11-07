"""Fetch day-ahead weather data from Open-Meteo and NASA POWER for S01.

Run once to download 2023-2024 weather data from two independent sources.
The output CSV is used by build_forecast_features() in data_quality.py.

Usage:
    python -m src.fetch_weather
    python -m src.fetch_weather --start 2023-01-01 --end 2024-12-31
    python -m src.fetch_weather --skip-nasa     # Open-Meteo only
    python -m src.fetch_weather --skip-clearsky  # skip pvlib clear-sky
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# S01 plant location (Nong Waeng, Khok Sung, Sa Kaeo Province)
LATITUDE = 13.82
LONGITUDE = 102.16
TIMEZONE = "Asia/Bangkok"

# ---------------------------------------------------------------------------
# Open-Meteo Historical Weather API
# ---------------------------------------------------------------------------
OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"

OPENMETEO_VARS = [
    # Solar radiation
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "sunshine_duration",
    # Temperature
    "temperature_2m",
    "dewpoint_2m",
    "apparent_temperature",
    # Humidity and atmospheric
    "relative_humidity_2m",
    "vapour_pressure_deficit",
    "pressure_msl",
    "precipitation",
    # Cloud cover (total + layers)
    "cloudcover",
    "cloudcover_low",
    "cloudcover_mid",
    "cloudcover_high",
    # Wind
    "windspeed_10m",
    "windgusts_10m",
]

OPENMETEO_RENAME = {
    "shortwave_radiation":       "fcst_ghi",
    "direct_radiation":          "fcst_direct_rad",
    "diffuse_radiation":         "fcst_diffuse_rad",
    "direct_normal_irradiance":  "fcst_dni",
    "sunshine_duration":         "fcst_sunshine_sec",
    "temperature_2m":            "fcst_temp",
    "dewpoint_2m":               "fcst_dewpoint",
    "apparent_temperature":      "fcst_apparent_temp",
    "relative_humidity_2m":      "fcst_humidity",
    "vapour_pressure_deficit":   "fcst_vpd",
    "pressure_msl":              "fcst_pressure",
    "precipitation":             "fcst_precip",
    "cloudcover":                "fcst_cloud",
    "cloudcover_low":            "fcst_cloud_low",
    "cloudcover_mid":            "fcst_cloud_mid",
    "cloudcover_high":           "fcst_cloud_high",
    "windspeed_10m":             "fcst_wind",
    "windgusts_10m":             "fcst_windgusts",
}

# ---------------------------------------------------------------------------
# NASA POWER Hourly API
# ---------------------------------------------------------------------------
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

NASA_PARAMS = [
    "ALLSKY_SFC_SW_DWN",
    "CLRSKY_SFC_SW_DWN",
    "ALLSKY_KT",
    "T2M",
    "RH2M",
    "WS2M",
]

NASA_RENAME = {
    "ALLSKY_SFC_SW_DWN": "nasa_ghi",
    "CLRSKY_SFC_SW_DWN": "nasa_clearsky_ghi",
    "ALLSKY_KT":         "nasa_clearness_index",
    "T2M":               "nasa_temp",
    "RH2M":              "nasa_humidity",
    "WS2M":              "nasa_wind",
}


def fetch_open_meteo(start_date: str, end_date: str,
                     chunk_months: int = 3) -> pd.DataFrame:
    """Fetch hourly weather from Open-Meteo Historical Weather API."""
    all_frames = []
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    while current <= end:
        chunk_end = min(
            current + pd.DateOffset(months=chunk_months) - pd.Timedelta(days=1),
            end,
        )
        print(f"  Open-Meteo: {current.date()} to {chunk_end.date()} ...")

        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": str(current.date()),
            "end_date": str(chunk_end.date()),
            "hourly": ",".join(OPENMETEO_VARS),
            "timezone": TIMEZONE,
        }

        response = requests.get(OPENMETEO_URL, params=params, timeout=120)
        response.raise_for_status()
        data = response.json()

        if "hourly" not in data:
            raise ValueError(
                f"No hourly data: {data.get('reason', 'unknown')}"
            )

        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
        all_frames.append(df)

        current = chunk_end + pd.Timedelta(days=1)
        time.sleep(0.5)

    result = pd.concat(all_frames).sort_index()
    result = result[~result.index.duplicated(keep="first")]
    result = result.rename(columns=OPENMETEO_RENAME)
    return result


def fetch_nasa_power(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly data from NASA POWER API (max ~1 year per request)."""
    all_frames = []
    start_year = pd.Timestamp(start_date).year
    end_year = pd.Timestamp(end_date).year

    for year in range(start_year, end_year + 1):
        year_start = max(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(start_date))
        year_end = min(pd.Timestamp(f"{year}-12-31"), pd.Timestamp(end_date))

        start_str = year_start.strftime("%Y%m%d")
        end_str = year_end.strftime("%Y%m%d")
        print(f"  NASA POWER: {year_start.date()} to {year_end.date()} ...")

        params = {
            "parameters": ",".join(NASA_PARAMS),
            "community": "RE",
            "longitude": LONGITUDE,
            "latitude": LATITUDE,
            "start": start_str,
            "end": end_str,
            "format": "JSON",
            "time-standard": "LST",
        }

        response = requests.get(NASA_POWER_URL, params=params, timeout=180)
        response.raise_for_status()
        data = response.json()

        if "properties" not in data or "parameter" not in data["properties"]:
            raise ValueError(f"Unexpected NASA POWER response for {year}")

        param_data = data["properties"]["parameter"]
        records = {}
        for param_name, values in param_data.items():
            for timestamp_key, value in values.items():
                if timestamp_key not in records:
                    records[timestamp_key] = {}
                records[timestamp_key][param_name] = (
                    value if value != -999.0 else np.nan
                )

        df = pd.DataFrame.from_dict(records, orient="index")
        df.index = pd.to_datetime(df.index, format="%Y%m%d%H")
        df = df.sort_index()
        all_frames.append(df)

        time.sleep(1)

    result = pd.concat(all_frames).sort_index()
    result = result[~result.index.duplicated(keep="first")]
    result.index = result.index.tz_localize(None)
    result = result.rename(columns=NASA_RENAME)
    return result


def resample_to_15min(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """Resample hourly data to 15-minute intervals via linear interpolation."""
    start = df_hourly.index.min()
    end = df_hourly.index.max().normalize() + pd.Timedelta(hours=23, minutes=45)
    full_index = pd.date_range(start, end, freq="15min", name="timestamp")

    df_15min = df_hourly.reindex(full_index)
    df_15min = df_15min.interpolate(method="time", limit=4)

    # Clip physical bounds
    rad_cols = [c for c in df_15min.columns
                if any(kw in c for kw in ["ghi", "rad", "dni", "clearsky"])]
    for col in rad_cols:
        df_15min[col] = df_15min[col].clip(lower=0)

    cloud_cols = [c for c in df_15min.columns if "cloud" in c]
    for col in cloud_cols:
        df_15min[col] = df_15min[col].clip(0, 100)

    hum_cols = [c for c in df_15min.columns if "humidity" in c]
    for col in hum_cols:
        df_15min[col] = df_15min[col].clip(0, 100)

    precip_cols = [c for c in df_15min.columns if "precip" in c]
    for col in precip_cols:
        df_15min[col] = df_15min[col].clip(lower=0)

    return df_15min


def compute_clear_sky_ghi(index: pd.DatetimeIndex) -> pd.Series | None:
    """Compute theoretical clear-sky GHI using pvlib if available."""
    try:
        import pvlib
    except ImportError:
        print("  pvlib not installed; skipping clear-sky computation")
        print("  Install with: pip install pvlib")
        return None

    location = pvlib.location.Location(
        LATITUDE, LONGITUDE, tz=TIMEZONE, altitude=50
    )
    clear_sky = location.get_clearsky(index, model="ineichen")
    return clear_sky["ghi"].rename("clearsky_ghi")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch weather data for S01 solar plant"
    )
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--output", default="data/forecast_weather_15min.csv")
    parser.add_argument("--skip-nasa", action="store_true",
                        help="Skip NASA POWER (Open-Meteo only)")
    parser.add_argument("--skip-clearsky", action="store_true",
                        help="Skip pvlib clear-sky computation")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Weather data fetch for S01 ({LATITUDE}N, {LONGITUDE}E)")
    print(f"Period: {args.start} to {args.end}")
    print()

    # Step 1: Open-Meteo
    print("Step 1: Fetching Open-Meteo Archive API (18 variables)")
    df_meteo = fetch_open_meteo(args.start, args.end)
    print(f"  Rows: {len(df_meteo)}, Columns: {len(df_meteo.columns)}")
    print()

    # Step 2: NASA POWER
    if not args.skip_nasa:
        print("Step 2: Fetching NASA POWER API (6 variables)")
        try:
            df_nasa = fetch_nasa_power(args.start, args.end)
            print(f"  Rows: {len(df_nasa)}, Columns: {len(df_nasa.columns)}")
            df_hourly = df_meteo.join(df_nasa, how="left")
        except Exception as e:
            print(f"  NASA POWER failed: {e}")
            print("  Continuing with Open-Meteo only.")
            df_hourly = df_meteo
    else:
        print("Step 2: Skipping NASA POWER (--skip-nasa)")
        df_hourly = df_meteo
    print()

    # Step 3: Resample
    print("Step 3: Resampling to 15-minute intervals")
    df_15min = resample_to_15min(df_hourly)
    print(f"  Rows: {len(df_15min)}")
    print()

    # Step 4: Clear-sky
    if not args.skip_clearsky:
        print("Step 4: Computing pvlib clear-sky GHI")
        clearsky = compute_clear_sky_ghi(df_15min.index)
        if clearsky is not None:
            df_15min["clearsky_ghi"] = clearsky
            safe_cs = clearsky.replace(0, np.nan)
            df_15min["clear_sky_index"] = (
                df_15min["fcst_ghi"] / safe_cs
            ).clip(0, 1.5)
            print("  clear_sky_index computed")
    else:
        print("Step 4: Skipping pvlib (--skip-clearsky)")
    print()

    # Step 5: Save
    print(f"Step 5: Saving to {output_path}")
    df_15min.to_csv(output_path)

    print()
    print("=" * 60)
    print(f"TOTAL COLUMNS: {len(df_15min.columns)}")
    print("=" * 60)
    for col in sorted(df_15min.columns):
        valid = df_15min[col].notna().sum()
        print(f"  {col:30s}  {valid:>6d} valid  ({valid/len(df_15min)*100:.1f}%)")
    print()
    print(f"Rows: {len(df_15min)} | Range: {df_15min.index.min()} to {df_15min.index.max()}")
    print()
    print("Done. Use in notebook:")
    print(f'  forecast_weather = load_forecast_weather("{args.output}")')


if __name__ == "__main__":
    main()
