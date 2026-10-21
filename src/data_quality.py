"""การตรวจสอบ completeness และ provenance ของข้อมูล S01"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from IPython.display import display
except ImportError:
    # ให้โมดูลทำงานได้แม้เรียกจาก Python script ที่ไม่มี IPython
    def display(value):
        print(value.to_string() if hasattr(value, "to_string") else value)


def fit_slot_median_imputer(df_train_raw, df_apply_raw, feature_columns):
    """Fit time-slot medians on train only, then apply them to another split."""

    imputer = SlotMedianImputer(feature_columns=list(feature_columns)).fit(df_train_raw)
    return imputer.transform(df_train_raw), imputer.transform(df_apply_raw), imputer.slot_medians_


WEATHER_COLUMNS = ['PVTEMP1', 'PVTEMP2', 'AMBTEMP', 'PYRANO1', 'PYRANO2', 'WINDSPEED']
REVIEW_DATES = ['2023-04-24', '2023-04-25', '2023-07-30',
                '2023-11-26', '2023-11-27', '2024-03-26', '2024-04-26']


def clean_sensor_observations(frame, output_dir=None):
    """Apply the existing sensor heuristics using only the current/past readings.

    A frozen reading is masked from its 12th consecutive observation onward.
    Appending future readings cannot change any previously cleaned value.
    Thresholds are study heuristics, not independently verified sensor faults.
    Genuine zero P and small negative measurements are retained.
    """
    df = frame.copy().sort_index()
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.has_duplicates:
        raise ValueError('Sensor cleaning requires unique datetime timestamps')
    rows = []
    df['flag_negative_P'] = df['P'] < 0
    rows.append(('Negative P', int(df['flag_negative_P'].sum()), 'Flag only; P retained'))
    df['flag_wind_stuck_60'] = df['WINDSPEED'].eq(60)
    df.loc[df['flag_wind_stuck_60'], 'WINDSPEED'] = np.nan
    rows.append(('WINDSPEED equals 60', int(df['flag_wind_stuck_60'].sum()), 'Wind set to missing'))
    wind_spike = df['WINDSPEED'].gt(20)
    df['flag_wind_spike'] = wind_spike
    df.loc[wind_spike, 'WINDSPEED'] = np.nan
    rows.append(('Other wind above 20 m/s', int(wind_spike.sum()), 'Wind set to missing'))
    df['flag_logger_dropout'] = df[WEATHER_COLUMNS].eq(0).all(axis=1)
    df.loc[df['flag_logger_dropout'], WEATHER_COLUMNS] = np.nan
    rows.append(('All weather sensors equal zero', int(df['flag_logger_dropout'].sum()), 'Weather set to missing'))
    df['flag_ambtemp_stuck'] = df['AMBTEMP'].eq(50)
    df.loc[df['flag_ambtemp_stuck'], 'AMBTEMP'] = np.nan
    rows.append(('AMBTEMP equals 50 C', int(df['flag_ambtemp_stuck'].sum()), 'Temperature set to missing'))

    consecutive = df.index.to_series().diff().eq(pd.Timedelta(minutes=5))
    def elapsed_run(series):
        same = series.eq(series.shift()) & consecutive & series.notna()
        groups = (~same).cumsum()
        return series.groupby(groups).cumcount() + 1
    stale = (elapsed_run(df['AMBTEMP']).ge(12)
             & elapsed_run(df['PVTEMP1']).ge(12)
             & df['AMBTEMP'].notna() & df['PVTEMP1'].notna()
             & df['AMBTEMP'].ne(0))
    df['flag_stale_data'] = stale
    df.loc[stale, WEATHER_COLUMNS] = np.nan
    rows.append(('Frozen temperature run from reading 12 onward', int(stale.sum()), 'Weather set to missing causally'))
    # Legacy dates remain review flags, rather than unsupported outage diagnoses.
    df['flag_review_date'] = df.index.normalize().isin(pd.to_datetime(REVIEW_DATES))
    summary = pd.DataFrame(rows, columns=['Rule', 'Rows affected', 'Handling'])
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_dir/'table_06_sensor_masking_summary.csv', index=False)
        rules = pd.DataFrame([
            ('WINDSPEED', '= 60', 'Retained study heuristic; confirm sensor code/units'),
            ('WINDSPEED', '> 20 m/s', 'Study threshold; sensitivity/site validation required'),
            ('All six weather sensors', '= 0', 'Possible logger issue; P is not masked'),
            ('AMBTEMP', '= 50 C', 'Retained study heuristic; confirm sensor range'),
            ('AMBTEMP and PVTEMP1', '12 identical consecutive 5-min readings',
             'Trailing-only detector; mask from reading 12, without backfilling'),
        ], columns=['Input', 'Threshold', 'Interpretation'])
        rules.to_csv(output_dir/'table_06_sensor_rule_definitions.csv', index=False)
    return df, summary


def daily_energy_kwh(power):
    """Energy for complete 96-slot days; incomplete daily totals remain missing."""
    power = power.replace([np.inf, -np.inf], np.nan)
    groups = power.groupby(power.index.normalize())
    return (groups.sum(min_count=96)*0.25).where(groups.count().eq(96))


def apply_day_quality_policy(frame, output_dir=None, min_valid_targets=80,
                             manual_exclusions=None):
    """Log every day's eligibility, with explicit reasons and observed evidence.

    The existing <80/96 completeness threshold is made an explicit study policy.
    Partial days above the threshold retain finite targets only. Legacy review
    dates with 96 values are retained unless a documented manual reason is given.
    This is an offline scoring policy, not an input to the forecast model.
    """
    manual_exclusions = {} if manual_exclusions is None else manual_exclusions
    reasons = {pd.Timestamp(k).normalize(): str(v).strip()
               for k, v in manual_exclusions.items()}
    if any(not v for v in reasons.values()):
        raise ValueError('Every manual exclusion requires a documented reason')
    df = frame.copy()
    log = pd.DataFrame({'valid_target_intervals': df['P'].notna().groupby(df.index.normalize()).sum()})
    log.index.name = 'date'
    log['complete_96'] = log['valid_target_intervals'].eq(96)
    log['legacy_review_date'] = log.index.isin(pd.to_datetime(REVIEW_DATES))
    log['manual_reason'] = [reasons.get(day, '') for day in log.index]
    log['excluded'] = log['valid_target_intervals'].lt(min_valid_targets) | log['manual_reason'].ne('')
    log['reason'] = [
        reason if reason else
        (f'Fewer than {min_valid_targets}/96 valid P targets ({int(n)}/96)'
         if n < min_valid_targets else
         ('Retained for review; no confirmed manual exclusion evidence'
          if review else
          ('Retained; missing target intervals will not be fitted/scored'
           if n < 96 else 'Retained; 96/96 valid P targets')))
        for n, review, reason in zip(log.valid_target_intervals, log.legacy_review_date, log.manual_reason)
    ]
    for flag in [c for c in df if c.startswith('flag_')]:
        log[flag+'_intervals'] = df[flag].fillna(False).astype(bool).groupby(df.index.normalize()).sum()
    df['excluded'] = df.index.normalize().map(log['excluded']).to_numpy(dtype=bool)
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        log.to_csv(Path(output_dir)/'table_06_day_quality_policy.csv')
    return df, log


def build_forecast_features(frame):
    """Create 17 features using D-1 or earlier values and known calendar fields."""
    df = frame.sort_index()
    if df.index.has_duplicates:
        raise ValueError('Feature timestamps must be unique')
    out = df[['P']].copy()
    for days in [1, 2, 7]:
        out[f'P_lag{days}'] = _calendar_day_lookup(df.P, df.index, days)
    history = pd.concat([_calendar_day_lookup(df.P, df.index, d) for d in range(1,8)], axis=1)
    out['P_roll7'] = _aggregate_with_min_count(history, 'mean')
    for name, raw in [('irr_lag1','irradiance'), ('pv_temp_lag1','pv_temp'),
                      ('amb_lag1','AMBTEMP'), ('wind_lag1','WINDSPEED')]:
        out[name] = _calendar_day_lookup(df[raw], df.index, 1)
    slot = df.index.hour*4 + df.index.minute//15
    for prefix, value, period in [('interval',slot,96), ('doy',df.index.dayofyear,365.25),
                                   ('month',df.index.month,12)]:
        out[prefix+'_sin'] = np.sin(2*np.pi*value/period)
        out[prefix+'_cos'] = np.cos(2*np.pi*value/period)
    out['dow'] = df.index.dayofweek
    energy = daily_energy_kwh(df.P)
    ehistory = pd.concat([energy.reindex(df.index.normalize()-pd.Timedelta(days=d)).set_axis(df.index)
                          for d in range(1,8)], axis=1)
    out['energy_roll7_mean'] = _aggregate_with_min_count(ehistory, 'mean')
    out['energy_roll7_std'] = _aggregate_with_min_count(ehistory, 'std')
    return out


class SlotMedianImputer(BaseEstimator, TransformerMixin):
    """Impute feature columns with time-slot medians learned from ``fit`` data only.

    This transformer is designed for an sklearn ``Pipeline``.  Therefore
    ``RandomizedSearchCV`` clones and fits a fresh copy inside every
    TimeSeriesSplit training fold; validation rows are only passed to
    ``transform`` and can never change the learned medians.
    """

    def __init__(self, feature_columns=None):
        self.feature_columns = feature_columns

    def _validate_frame(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SlotMedianImputer expects a pandas DataFrame with a DatetimeIndex")
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("SlotMedianImputer requires a pandas DatetimeIndex")
        return X.copy()

    def fit(self, X, y=None):
        train = self._validate_frame(X)
        self.feature_columns_ = list(self.feature_columns or train.columns)
        if 'P' in self.feature_columns_:
            raise ValueError('P must never be imputed')
        missing_columns = sorted(set(self.feature_columns_) - set(train.columns))
        if missing_columns:
            raise KeyError(f"Missing imputation columns: {missing_columns}")

        # คำนวณสถิติจากข้อมูลที่ sklearn ส่งเข้า fit เท่านั้น
        # ใน RandomizedSearchCV นี่คือ training fold ไม่ใช่ validation fold
        train["_interval_idx"] = (
            train.index.hour * 4 + train.index.minute // 15
        )
        self.slot_medians_ = train.groupby("_interval_idx")[
            self.feature_columns_
        ].median()
        # fallback นี้ยังมาจาก training fold เดิมเท่านั้น กรณี slot ใดไม่มีข้อมูล
        self.global_medians_ = train[self.feature_columns_].median()
        # An entirely absent sensor contributes no information. A fixed zero
        # fallback keeps the schema stable without looking at validation data.
        self.global_medians_ = self.global_medians_.fillna(0.0)
        return self

    def transform(self, X):
        if not hasattr(self, "slot_medians_"):
            raise RuntimeError("SlotMedianImputer must be fitted before transform")

        apply = self._validate_frame(X)
        missing_columns = sorted(set(self.feature_columns_) - set(apply.columns))
        if missing_columns:
            raise KeyError(f"Missing imputation columns: {missing_columns}")
        apply["_interval_idx"] = (
            apply.index.hour * 4 + apply.index.minute // 15
        )
        for column in self.feature_columns_:
            apply[column] = apply[column].fillna(
                apply["_interval_idx"].map(self.slot_medians_[column])
            )
            apply[column] = apply[column].fillna(self.global_medians_[column])
        return apply.drop(columns=["_interval_idx"])


def validate_forecast_observations(y_true, y_pred, index=None, label="forecast"):
    """Require finite, aligned observations so every model uses the same test rows."""

    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if index is None:
        index = pd.RangeIndex(len(y_true))
    else:
        index = pd.DatetimeIndex(index)
    if len(y_true) != len(y_pred) or len(y_true) != len(index):
        raise ValueError(
            f"{label}: y_true, y_pred, and index must have the same length"
        )
    if index.has_duplicates:
        raise ValueError(f"{label}: forecast index contains duplicate timestamps")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError(
            f"{label}: metrics require finite y_true and y_pred; "
            "filter the common evaluation rows before scoring"
        )
    if len(y_true) == 0:
        raise ValueError(f"{label}: no observations are available for evaluation")
    return y_true, y_pred, index


def summarize_forecast_day_completeness(
    index,
    target_values=None,
    output_dir=None,
    file_name="table_14_forecast_day_completeness.csv",
    expected_intervals=96,
    frequency="15min",
    expected_dates=None,
):
    """Report complete and incomplete forecast days from index and optional target values."""

    raw_index = pd.DatetimeIndex(index)
    order = np.argsort(raw_index.asi8)
    index = raw_index[order]
    if target_values is None:
        values = pd.Series(np.nan, index=index, dtype=float)
        has_target = False
    else:
        raw_values = np.asarray(target_values, dtype=float).reshape(-1)
        if len(raw_values) != len(raw_index):
            raise ValueError("target_values must have the same length as index")
        values = pd.Series(raw_values[order], index=index)
        has_target = True

    rows = []
    dates = (pd.Index(index.normalize()).unique().sort_values()
             if expected_dates is None else pd.DatetimeIndex(expected_dates).normalize().unique().sort_values())
    for date in dates:
        day_mask = index.normalize() == date
        observed = index[day_mask].drop_duplicates()
        expected = pd.date_range(date, periods=expected_intervals, freq=frequency)
        valid_target = int(np.isfinite(values.loc[day_mask]).sum()) if has_target else len(observed)
        rows.append(
            {
                "date": date,
                "expected_intervals": expected_intervals,
                "observed_unique_intervals": len(observed),
                "valid_target_intervals": valid_target,
                "missing_index_intervals": len(expected.difference(observed)),
                "duplicate_rows": int(day_mask.sum() - len(observed)),
                "index_complete": len(observed) == expected_intervals and len(expected.difference(observed)) == 0,
                "target_complete": (
                    len(observed) == expected_intervals
                    and len(expected.difference(observed)) == 0
                    and int(day_mask.sum() - len(observed)) == 0
                    and valid_target == expected_intervals
                ),
                "forecast_day_status": (
                    "complete 96/96 target"
                    if len(observed) == expected_intervals
                    and len(expected.difference(observed)) == 0
                    and int(day_mask.sum() - len(observed)) == 0
                    and valid_target == expected_intervals
                    else "incomplete target; excluded rows are not scored"
                ),
            }
        )

    summary = pd.DataFrame(rows)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_dir / file_name, index=False)
    return summary


def _source_sheet_rows(excel_files):
    """อ่านจำนวน timestamp จาก worksheet จริงเพื่อแยก source gap กับ loader gap"""

    source_rows = []
    for file_path in excel_files:
        match = re.search(r"S01-(\d{2})-(\d{2})", Path(file_path).stem)
        if match is None:
            continue
        file_month = int(match.group(1))
        file_year = 2000 + int(match.group(2))
        xls = pd.ExcelFile(file_path, engine="openpyxl")
        try:
            for sheet in xls.sheet_names:
                try:
                    day = int(str(sheet).strip())
                    source_date = pd.Timestamp(file_year, file_month, day)
                except (ValueError, TypeError):
                    continue

                raw_sheet = xls.parse(sheet, header=None)
                # นับเฉพาะแถวที่มีเวลา เพื่อไม่รวมแถวว่างท้าย worksheet
                data_rows = (
                    raw_sheet.iloc[3:]
                    if raw_sheet.shape[0] >= 4 and raw_sheet.shape[1] >= 2
                    else pd.DataFrame()
                )
                valid_rows = (
                    data_rows[data_rows.iloc[:, 1].notna()]
                    if not data_rows.empty
                    else data_rows
                )
                source_5min_rows = int(len(valid_rows))
                # ตรวจ P จาก worksheet ต้นทางโดยตรง เพื่อแยก source gap กับ loader gap
                source_p_missing = (
                    int(pd.to_numeric(valid_rows.iloc[:, 8], errors="coerce").isna().sum())
                    if valid_rows.shape[1] >= 9
                    else source_5min_rows
                )
                source_rows.append(
                    {
                        "date": source_date.normalize(),
                        "source_sheet_present": True,
                        "source_5min_rows": source_5min_rows,
                        "source_p_missing_5min": source_p_missing,
                    }
                )
        finally:
            xls.close()

    if not source_rows:
        raise ValueError("ไม่พบ worksheet ที่ใช้ทำ source audit")

    return (
        pd.DataFrame(source_rows)
        .groupby("date")
        .agg(
            source_sheet_present=("source_sheet_present", "any"),
            source_5min_rows=("source_5min_rows", "sum"),
            source_p_missing_5min=("source_p_missing_5min", "sum"),
        )
    )


def run_completeness_audit(
    df_raw,
    daily_completeness,
    coverage,
    excel_files,
    output_dir,
    study_start="2023-01-01",
    study_end="2024-12-31",
):
    """สร้างตาราง raw/source/loader/target completeness และวันที่ที่มี gap"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # สร้าง raw daily summary จากข้อมูลที่ loader ส่งต่อให้ notebook
    expected_dates = pd.date_range(
        study_start,
        study_end,
        freq="D",
    )
    raw_daily = (
        df_raw.assign(date=df_raw["timestamp"].dt.normalize())
        .groupby("date")
        .agg(
            loaded_5min_rows=("timestamp", "size"),
            loaded_unique_timestamps=("timestamp", "nunique"),
            raw_p_missing_5min=("P", lambda s: int(s.isna().sum())),
        )
    )
    source_daily = _source_sheet_rows(excel_files)

    # รวม source sheet, loader rows และ valid target intervals ใน index เดียวกัน
    audit = pd.DataFrame(index=expected_dates)
    audit.index.name = "date"
    audit = audit.join(source_daily).join(raw_daily)
    daily = daily_completeness.copy()
    daily.index = pd.to_datetime(daily.index)
    audit = audit.join(daily)
    audit["source_sheet_present"] = audit["source_sheet_present"].astype("boolean").fillna(False).astype(bool)
    for column in [
        "source_5min_rows",
        "loaded_5min_rows",
        "loaded_unique_timestamps",
        "raw_p_missing_5min",
        "source_p_missing_5min",
        "valid_intervals",
    ]:
        audit[column] = audit[column].fillna(0).astype(int)
    audit["complete_96"] = audit["valid_intervals"] == 96

    # คอลัมน์นี้บอกต้นเหตุของ gap ระดับวันแบบตรวจสอบย้อนกลับได้
    audit["source_loader_status"] = np.select(
        [
            ~audit["source_sheet_present"],
            audit["loaded_5min_rows"] < audit["source_5min_rows"],
            audit["loaded_5min_rows"] != audit["loaded_unique_timestamps"],
            audit["loaded_5min_rows"] > audit["source_5min_rows"],
            audit["source_5min_rows"] < 288,
            audit["raw_p_missing_5min"] > audit["source_p_missing_5min"],
            audit["source_p_missing_5min"] > 0,
        ],
        [
            "missing source worksheet",
            "loader dropped rows",
            "duplicate loaded timestamps",
            "more loaded rows than source rows",
            "incomplete source worksheet",
            "loader introduced P missing",
            "source P values missing",
        ],
        default="loaded as expected",
    )
    audit["raw_row_coverage_%"] = (
        audit["loaded_unique_timestamps"] / 288 * 100
    ).round(2)
    audit["timestamp_row_difference"] = (
        audit["loaded_5min_rows"] - audit["source_5min_rows"]
    )
    audit["P_missing_difference"] = (
        audit["raw_p_missing_5min"] - audit["source_p_missing_5min"]
    )
    audit["target_interval_coverage_%"] = (
        audit["valid_intervals"] / 96 * 100
    ).round(2)
    audit["target_status"] = np.where(
        audit["complete_96"],
        "complete target day",
        "incomplete P; intervals are excluded from loss",
    )

    # สรุป monthly coverage ของ target จริง แยกจาก raw row coverage
    monthly_target = (
        audit.assign(year=audit.index.year, month=audit.index.month)
        .groupby(["year", "month"])
        .agg(
            calendar_days=("valid_intervals", "size"),
            source_sheet_days=("source_sheet_present", "sum"),
            raw_loaded_days=("loaded_5min_rows", lambda s: int((s > 0).sum())),
            complete_target_days=("complete_96", "sum"),
            incomplete_target_days=("complete_96", lambda s: int((~s).sum())),
            valid_target_intervals=("valid_intervals", "sum"),
        )
        .reset_index()
    )
    monthly_target["expected_target_intervals"] = monthly_target["calendar_days"] * 96
    monthly_target["target_day_coverage_%"] = (
        monthly_target["complete_target_days"]
        / monthly_target["calendar_days"]
        * 100
    ).round(2)
    monthly_target["target_interval_coverage_%"] = (
        monthly_target["valid_target_intervals"]
        / monthly_target["expected_target_intervals"]
        * 100
    ).round(2)

    # รวม raw coverage เดิมกับ target coverage ใหม่ เพื่อไม่ให้รายงาน 100% แบบกำกวม
    coverage_report = coverage.drop(columns=['calendar_days'], errors='ignore').rename(
        columns={"coverage_%": "raw_row_coverage_%"}
    ).merge(monthly_target, on=["year", "month"], how="left")
    if "raw_row_coverage_%" not in coverage_report.columns:
        # fallback สำหรับ coverage table รุ่นเก่าที่เก็บ rows แต่ยังไม่มีเปอร์เซ็นต์
        coverage_report["raw_row_coverage_%"] = (
            coverage_report["rows"]
            / (coverage_report["calendar_days"] * 288)
            * 100
        ).round(2)
    coverage_report.to_csv(output_dir / "table_01_monthly_data_coverage.csv", index=False)
    audit.to_csv(output_dir / "table_01_source_loader_audit.csv")
    audit.loc[~audit["complete_96"]].to_csv(
        output_dir / "table_01_missing_target_dates.csv"
    )

    print("Corrected target completeness by month")
    display(monthly_target)
    print("Source/loader status counts")
    display(audit["source_loader_status"].value_counts().to_frame("days"))
    print("Missing target dates saved to table_01_missing_target_dates.csv")
    return audit, monthly_target, coverage_report


def _calendar_day_lookup(series, target_index, days_back):
    """lookup ค่าจากวันก่อนหน้าโดยใช้ timestamp จริง"""

    values = series.reindex(target_index - pd.Timedelta(days=days_back))
    values.index = target_index
    return values


def _matches(left, right):
    """เปรียบเทียบ Series ที่อาจมี NaN โดยถือ NaN ตรงกัน"""

    return bool(
        np.allclose(
            left.to_numpy(dtype=float),
            right.to_numpy(dtype=float),
            equal_nan=True,
        )
    )


def _aggregate_with_min_count(frame, method, min_count=3):
    """คำนวณ aggregate แบบกำหนดจำนวนค่าขั้นต่ำให้รองรับ pandas หลายรุ่น"""

    counts = frame.notna().sum(axis=1)
    values = getattr(frame, method)(axis=1, skipna=True)
    return values.where(counts >= min_count)


def run_feature_availability_audit(
    df_full_imputed,
    df_features,
    feature_names,
    output_dir,
):
    """ตรวจ provenance ของ feature และยืนยันว่าไม่มี source จากวัน forecast"""

    output_dir = Path(output_dir)
    target_index = df_features.index
    target_days = target_index.normalize()
    # Forecast for day D is issued at 23:55 on day D-1
    forecast_issue_deadline = target_days - pd.Timedelta(minutes=5)
    rows = []

    exact_features = {
        "P_lag1": ("P", 1),
        "P_lag2": ("P", 2),
        "P_lag7": ("P", 7),
        "irr_lag1": ("irradiance", 1),
        "pv_temp_lag1": ("pv_temp", 1),
        "amb_lag1": ("AMBTEMP", 1),
        "wind_lag1": ("WINDSPEED", 1),
    }
    for feature, (source_column, lag_days) in exact_features.items():
        expected = _calendar_day_lookup(
            df_full_imputed[source_column], target_index, lag_days
        )
        value_check = _matches(df_features[feature], expected)
        source_timestamps = target_index - pd.Timedelta(days=lag_days)
        source_before_forecast = bool(
            (source_timestamps <= forecast_issue_deadline).all()
        )
        rows.append(
            {
                "Feature": feature,
                "Source": f"{source_column}, day D-{lag_days}",
                "Max source lag (days)": lag_days,
                "Available before forecast": source_before_forecast,
                "Forecast issue deadline": "D-1 23:55",
                "Value provenance check": "PASS" if value_check else "FAIL",
            }
        )

    # ตรวจ rolling features เทียบกับค่าที่สร้างจากวันก่อนหน้า 1-7 เท่านั้น
    p_history = pd.concat(
        [
            _calendar_day_lookup(df_full_imputed["P"], target_index, d)
            for d in range(1, 8)
        ],
        axis=1,
    )
    energy = daily_energy_kwh(df_full_imputed["P"])
    energy_history = pd.concat(
        [
            energy.reindex(
                target_index.normalize() - pd.Timedelta(days=d)
            ).set_axis(target_index)
            for d in range(1, 8)
        ],
        axis=1,
    )
    rolling_checks = {
        "P_roll7": (
            "P, same interval during days D-1 to D-7",
            _aggregate_with_min_count(p_history, "mean"),
        ),
        "energy_roll7_mean": (
            "daily P energy during days D-1 to D-7",
            _aggregate_with_min_count(energy_history, "mean"),
        ),
        "energy_roll7_std": (
            "daily P energy during days D-1 to D-7",
            _aggregate_with_min_count(energy_history, "std"),
        ),
    }
    for feature, (source, expected) in rolling_checks.items():
        rows.append(
            {
                "Feature": feature,
                "Source": source,
                "Max source lag (days)": 7,
                "Available before forecast": bool(
                    ((target_index - pd.Timedelta(days=1)) <= forecast_issue_deadline).all()
                ),
                "Forecast issue deadline": "D-1 23:55",
                "Value provenance check": (
                    "PASS" if _matches(df_features[feature], expected) else "FAIL"
                ),
            }
        )

    calendar = {
        'interval_sin': np.sin(2*np.pi*(target_index.hour*4+target_index.minute//15)/96),
        'interval_cos': np.cos(2*np.pi*(target_index.hour*4+target_index.minute//15)/96),
        'doy_sin': np.sin(2*np.pi*target_index.dayofyear/365.25),
        'doy_cos': np.cos(2*np.pi*target_index.dayofyear/365.25),
        'month_sin': np.sin(2*np.pi*target_index.month/12),
        'month_cos': np.cos(2*np.pi*target_index.month/12),
        'dow': target_index.dayofweek,
    }
    remaining = set(feature_names) - set(exact_features) - set(rolling_checks)
    unknown = remaining - set(calendar)
    if unknown:
        raise ValueError(f'Features without an availability definition: {sorted(unknown)}')
    for feature in sorted(remaining):
        rows.append(
            {
                "Feature": feature,
                "Source": "calendar known at forecast time",
                "Max source lag (days)": 0,
                "Available before forecast": True,
                "Forecast issue deadline": "D-1 23:55",
                "Value provenance check": ('PASS' if np.allclose(
                    df_features[feature].to_numpy(), calendar[feature], equal_nan=True
                ) else 'FAIL'),
            }
        )

    audit = pd.DataFrame(rows).sort_values("Feature").reset_index(drop=True)
    assert audit["Available before forecast"].all(), (
        "พบ feature ที่ใช้ข้อมูลหลังเวลา forecast"
    )
    assert (audit["Value provenance check"] == "PASS").all(), (
        "พบ feature provenance ไม่ตรงกับ calendar-day definition"
    )
    audit.to_csv(output_dir / "table_07_feature_availability_audit.csv", index=False)
    print("Feature availability audit: PASS")
    display(audit)
    return audit
