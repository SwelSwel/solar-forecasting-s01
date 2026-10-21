"""โมดูลสำหรับโมเดลขั้นต่อไปของโครงการพยากรณ์ไฟฟ้า S01

โค้ดในไฟล์นี้ต่อจาก feature matrix ที่ notebook สร้างไว้แล้ว โดยไม่ทำ data
cleaning ซ้ำ เพื่อให้ notebook อ่านง่ายและสามารถเรียกใช้ฟังก์ชันเดิมได้อีกครั้ง
ในอนาคต
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_quality import (
    SlotMedianImputer,
    fit_slot_median_imputer,
    summarize_forecast_day_completeness,
    validate_forecast_observations,
)


class CalendarDayTimeSeriesSplit(BaseCrossValidator):
    """Expanding CV with whole calendar dates and a calendar-day embargo."""
    def __init__(self, n_splits=5, gap_days=1):
        self.n_splits = n_splits
        self.gap_days = gap_days

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError('Calendar CV requires a DatetimeIndex')
        if not X.index.is_monotonic_increasing or X.index.has_duplicates:
            raise ValueError('Calendar CV requires sorted, unique timestamps')
        if self.gap_days < 0:
            raise ValueError('gap_days must be nonnegative')
        row_dates = X.index.normalize()
        dates = row_dates.unique().sort_values()
        for train_days, valid_days in TimeSeriesSplit(n_splits=self.n_splits).split(dates):
            validation_dates = dates[valid_days]
            deadline = validation_dates[0] - pd.Timedelta(days=self.gap_days)
            training_dates = dates[train_days]
            training_dates = training_dates[training_dates < deadline]
            if len(training_dates) == 0:
                raise ValueError('Insufficient training dates after the calendar embargo')
            yield (np.flatnonzero(row_dates.isin(training_dates)),
                   np.flatnonzero(row_dates.isin(validation_dates)))


POWER_HISTORY = ['P_lag1','P_lag2','P_lag7','P_roll7','energy_roll7_mean','energy_roll7_std']
BASELINE_NAMES = ['Persistence (t-1)', 'Persistence (t-7)', 'Historical Average']


def postprocess_predictions(values, capacity_kw=6000):
    """One physical-range policy for all predictions; observed targets stay intact."""
    return np.clip(np.asarray(values, dtype=float), 0, capacity_kw)


def baseline_predictions(train_raw, test_features, capacity_kw=6000):
    """Training-only monthly/slot profile with a training-only slot fallback."""
    if train_raw.index.max().normalize() >= test_features.index.min().normalize():
        raise ValueError('Baseline training dates must precede all test dates')
    train = train_raw.dropna(subset=['P']).copy()
    train['month'] = train.index.month
    train['slot'] = train.index.hour*4 + train.index.minute//15
    profile = train.groupby(['month','slot'])['P'].mean()
    fallback = train.groupby('slot')['P'].mean()
    predictions = {
        BASELINE_NAMES[0]: test_features['P_lag1'].to_numpy(),
        BASELINE_NAMES[1]: test_features['P_lag7'].to_numpy(),
        BASELINE_NAMES[2]: np.array([
            profile.get((m,s), fallback.get(s, np.nan))
            for m,s in zip(test_features.index.month,
                           test_features.index.hour*4 + test_features.index.minute//15)
        ]),
    }
    return {name:postprocess_predictions(pred,capacity_kw) for name,pred in predictions.items()}


def evaluate_baselines(train_raw, test_features, capacity_kw=6000):
    predictions = baseline_predictions(train_raw, test_features, capacity_kw)
    rows = []
    for name, pred in predictions.items():
        rows.extend(_evaluate(test_features.P, pred, test_features.index, name, capacity_kw))
    return _add_skill_score(pd.DataFrame(rows)), predictions


def _feature_frame(raw, build_features, feature_names):
    # Leave weather NaN for training-only pipeline imputation. Never impute P.
    required = [c for c in POWER_HISTORY if c in feature_names] + ['P']
    frame = build_features(raw).replace([np.inf,-np.inf],np.nan)
    return frame.dropna(subset=required)


def _monthly_windows(raw, feature_names, build_features, min_train_months=6):
    months = pd.period_range(raw.index.min().to_period('M'), raw.index.max().to_period('M'), freq='M')
    features = _feature_frame(raw, build_features, feature_names)
    for period in months[min_train_months:]:
        train_raw = raw.loc[raw.index < period.start_time]
        train_feat = features.loc[features.index < period.start_time]
        test_feat = features.loc[(features.index >= period.start_time) & (features.index <= period.end_time)]
        if not train_feat.empty and not test_feat.empty:
            yield period, train_raw, train_feat, test_feat


def _fit_xgb_search(X, y, impute_columns, random_state=42, n_iter=30, cv=None):
    cv = CalendarDayTimeSeriesSplit() if cv is None else cv
    if not isinstance(cv, CalendarDayTimeSeriesSplit):
        raise TypeError('Use CalendarDayTimeSeriesSplit for a day-ahead CV experiment')
    parameters = {
        'n_estimators':[100,200,300,500], 'max_depth':[3,4,5,6,8],
        'learning_rate':[0.01,0.05,0.1,0.2], 'subsample':[0.6,0.7,0.8,0.9,1.0],
        'colsample_bytree':[0.6,0.7,0.8,0.9,1.0], 'min_child_weight':[1,3,5],
    }
    pipeline = make_fold_safe_preprocessing_pipeline(_new_xgb(random_state=random_state), impute_columns)
    search = RandomizedSearchCV(pipeline,
        {f'model__{k}':v for k,v in parameters.items()}, n_iter=n_iter, cv=cv,
        scoring='neg_root_mean_squared_error', random_state=random_state,
        n_jobs=1, verbose=1, error_score='raise')
    search.fit(X,y)
    return search


def _cv_date_report(X, cv):
    rows=[]
    for fold,(tr,va) in enumerate(cv.split(X),1):
        train_dates=X.index[tr].normalize().unique()
        valid_dates=X.index[va].normalize().unique()
        assert train_dates.intersection(valid_dates).empty
        rows.append({'Fold':fold,'Train start':X.index[tr].min(), 'Train end':X.index[tr].max(),
                     'Validation start':X.index[va].min(), 'Validation end':X.index[va].max(),
                     'Train dates':len(train_dates), 'Validation dates':len(valid_dates),
                     'Embargo calendar days':cv.gap_days})
    return pd.DataFrame(rows)


def make_fold_safe_preprocessing_pipeline(estimator, feature_columns):
    """Wrap an estimator so preprocessing is fitted inside each CV fold."""

    return Pipeline(
        [
            # imputer นี้จะถูก clone และ fit ใหม่โดย TimeSeriesSplit ทุก fold
            ("imputer", SlotMedianImputer(feature_columns=list(feature_columns))),
            ("model", estimator),
        ]
    )


def _evaluate(y_true, y_pred, index, label, capacity_kw=6000):
    """คืน metric ทั้งช่วง 24 ชั่วโมงและช่วงกลางวัน 06:00-18:00"""

    y_true, y_pred, index = validate_forecast_observations(
        y_true, y_pred, index=index, label=label
    )

    def metric_for_mask(mask, metric_label):
        true_part = y_true[mask]
        pred_part = y_pred[mask]
        rmse = np.sqrt(mean_squared_error(true_part, pred_part))
        return {
            "Model": metric_label,
            "N": int(mask.sum()),
            "MAE": round(mean_absolute_error(true_part, pred_part), 2),
            "RMSE": round(rmse, 2),
            "nRMSE": round(rmse / capacity_kw, 4),
            "R2": round(r2_score(true_part, pred_part), 4),
        }

    full_mask = np.ones(len(index), dtype=bool)
    daytime_mask = (index.hour >= 6) & (index.hour < 18)
    return (
        metric_for_mask(full_mask, f"{label} (24h)"),
        metric_for_mask(daytime_mask, f"{label} (06-18)"),
    )


def _add_skill_score(results_df):
    """คำนวณ skill score ใหม่ให้ทุกโมเดล โดยเทียบกับ persistence t-1"""

    results_df = results_df.copy()
    if "N" in results_df.columns and results_df["N"].notna().all():
        for period in ("(24h)", "(06-18)"):
            counts = results_df.loc[
                results_df["Model"].str.endswith(period), "N"
            ].unique()
            if len(counts) != 1:
                raise ValueError(
                    f"Metrics are not comparable for {period}: observation counts={counts}"
                )
    reference_24h = results_df.loc[
        results_df["Model"] == "Persistence (t-1) (24h)", "MAE"
    ].iloc[0]
    reference_day = results_df.loc[
        results_df["Model"] == "Persistence (t-1) (06-18)", "MAE"
    ].iloc[0]

    results_df["Skill"] = results_df.apply(
        lambda row: round(
            1 - row["MAE"] / (
                reference_day if "(06-18)" in row["Model"] else reference_24h
            ),
            4,
        ),
        axis=1,
    )
    return results_df


def _new_xgb(params=None, random_state=42):
    """สร้าง XGBoost โดย import ตอนเรียกใช้ เพื่อให้ Ridge ยังใช้งานได้ถ้า package ขาด"""

    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover - ขึ้นกับ environment ของผู้ใช้
        raise ImportError(
            "ต้องติดตั้ง xgboost ก่อนรันส่วน XGBoost: pip install xgboost"
        ) from exc

    model_params = {} if params is None else dict(params)
    model_params.setdefault("objective", "reg:squarederror")
    model_params.setdefault("random_state", random_state)
    model_params.setdefault("n_jobs", -1)
    return XGBRegressor(**model_params)


def fit_advanced_models(
    X_train,
    y_train,
    X_test,
    y_test,
    test_index,
    feature_names: Iterable[str],
    baseline_results: pd.DataFrame,
    output_dir: Path,
    tscv=None,
    impute_feature_columns: Iterable[str] | None = None,
    plant_capacity_kw=6000,
    random_state=42,
    search_n_iter=30,
):
    """ฝึก Ridge, XGBoost, ablation และ prediction interval แล้วบันทึกผลลัพธ์"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = list(feature_names)
    X_train = X_train.copy()
    X_test = X_test.copy()
    if not pd.DatetimeIndex(test_index).equals(X_test.index):
        raise ValueError('test_index must match the feature timestamps exactly')
    for X,y in [(X_train,y_train),(X_test,y_test)]:
        if hasattr(y,'index') and not X.index.equals(y.index):
            raise ValueError('Feature and target timestamps are misaligned')
    # ตรวจสอบลำดับเวลาและการแยก hold-out ก่อนเริ่มฝึกโมเดล
    # เพื่อป้องกันการส่งข้อมูลอนาคตหรือแถวซ้ำระหว่าง train/test โดยไม่ตั้งใจ
    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_train and X_test must be pandas DataFrames")
    if not isinstance(X_train.index, pd.DatetimeIndex) or not isinstance(
        X_test.index, pd.DatetimeIndex
    ):
        raise TypeError("X_train and X_test must use a pandas DatetimeIndex")
    if not X_train.index.is_monotonic_increasing or not X_test.index.is_monotonic_increasing:
        raise ValueError("Training and test observations must be in chronological order")
    if X_train.index.intersection(X_test.index).size:
        raise ValueError("Training and hold-out test observations must not overlap")
    if X_train.index.max().normalize() >= X_test.index.min().normalize():
        raise ValueError('Hold-out training dates must be strictly before test dates')
    if impute_feature_columns is None:
        impute_feature_columns = list(feature_names)
    else:
        impute_feature_columns = list(impute_feature_columns)
    missing_imputation_columns = sorted(
        set(impute_feature_columns) - set(X_train.columns)
    )
    if missing_imputation_columns:
        raise KeyError(
            "ไม่พบ feature สำหรับ fold-safe imputation: "
            f"{missing_imputation_columns}"
        )

    # ทั้ง imputer และ scaler อยู่ใน pipeline เพื่อ fit จาก train เท่านั้น
    # และให้ CV clone pipeline ใหม่ในทุก TimeSeriesSplit fold
    # เพิ่ม scaler ใน pipeline ตามลำดับที่ถูกต้อง: impute -> scale -> Ridge
    ridge_pipeline = Pipeline(
        [
            ("imputer", SlotMedianImputer(feature_columns=impute_feature_columns)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    ridge_pipeline.fit(X_train, y_train)
    scaler = ridge_pipeline.named_steps["scaler"]
    model_ridge = ridge_pipeline.named_steps["model"]
    y_pred_ridge = np.clip(
        ridge_pipeline.predict(X_test), 0, plant_capacity_kw
    )
    ridge_full, ridge_day = _evaluate(
        y_test, y_pred_ridge, test_index, "Ridge Regression", plant_capacity_kw
    )

    # แสดง coefficient ที่มีขนาดมากที่สุดเพื่ออธิบายว่า Ridge ใช้ feature ใดมาก
    coef_df = pd.DataFrame(
        {"Feature": feature_names, "Coefficient": model_ridge.coef_}
    )
    coef_df["Abs"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Abs", ascending=False).head(10)
    coef_df.to_csv(output_dir / "table_10_ridge_top_coefficients.csv", index=False)

    tscv = CalendarDayTimeSeriesSplit() if tscv is None else tscv
    search = _fit_xgb_search(X_train, y_train, impute_feature_columns,
                             random_state, search_n_iter, tscv)
    _cv_date_report(X_train, tscv).to_csv(
        output_dir/'table_11_cv_date_boundaries.csv', index=False)
    model_xgb = search.best_estimator_
    y_pred_xgb = np.clip(model_xgb.predict(X_test), 0, plant_capacity_kw)
    xgb_full, xgb_day = _evaluate(
        y_test, y_pred_xgb, test_index, "XGBoost", plant_capacity_kw
    )

    best_params = {
        name.removeprefix("model__"): value
        for name, value in search.best_params_.items()
    }
    print("Best XGBoost parameters:")
    print(best_params)
    print(f"Best CV RMSE: {-search.best_score_:.2f}")

    # เก็บผลการค้นหาไว้ตรวจสอบได้ว่ามีการ tuning จริง ไม่ใช่รายงานเฉพาะค่าที่ดีที่สุด
    cv_results = (
        pd.DataFrame(search.cv_results_)
        .sort_values("rank_test_score")
        .reset_index(drop=True)
    )
    # เก็บชื่อคอลัมน์ hyperparameter แบบเดิมควบคู่กับชื่อ model__ ของ Pipeline
    for name in best_params:
        source = f"param_model__{name}"
        if source in cv_results:
            cv_results[f"param_{name}"] = cv_results[source]
    cv_results.to_csv(output_dir / "table_11_xgboost_random_search.csv", index=False)

    xgb_regressor = model_xgb.named_steps["model"]
    importances = pd.Series(xgb_regressor.feature_importances_, index=feature_names)
    top15 = importances.nlargest(15).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    top15.plot.barh(ax=ax)
    ax.set_xlabel("Feature Importance")
    ax.set_title("XGBoost Top 15 Feature Importances")
    plt.tight_layout()
    fig.savefig(output_dir / "figure_11_xgboost_feature_importance.png", dpi=150)
    plt.show()
    plt.close(fig)

    # Keep only the six simple-baseline rows even if this cell is reexecuted.
    allowed = [f'{name} {period}' for name in BASELINE_NAMES for period in ['(24h)','(06-18)']]
    baseline_results = baseline_results.loc[baseline_results.Model.isin(allowed)].drop_duplicates('Model',keep='last')
    if set(baseline_results.Model) != set(allowed):
        raise ValueError('Recompute the three baselines on the common test observations first')
    advanced_rows = [ridge_full, ridge_day, xgb_full, xgb_day]

    # เปรียบเทียบกลุ่ม feature โดยใช้ hyperparameters ชุดเดียวกับ XGBoost ตัวเต็ม
    group1 = ["P_lag1", "P_lag2", "P_lag7", "P_roll7"]
    group2 = group1 + ["irr_lag1", "pv_temp_lag1", "amb_lag1", "wind_lag1"]
    for label, columns in [
        ("XGBoost (lags only)", group1),
        ("XGBoost (lags+weather)", group2),
    ]:
        missing = sorted(set(columns) - set(X_train.columns))
        if missing:
            raise KeyError(f"ไม่พบ feature สำหรับ ablation: {missing}")
        ablation_impute_columns = [
            column for column in impute_feature_columns if column in columns
        ]
        ablation_model = make_fold_safe_preprocessing_pipeline(
            _new_xgb(best_params, random_state=random_state),
            ablation_impute_columns,
        )
        ablation_model.fit(X_train[columns], y_train)
        prediction = np.clip(
            ablation_model.predict(X_test[columns]), 0, plant_capacity_kw
        )
        full, day = _evaluate(
            y_test, prediction, test_index, label, plant_capacity_kw
        )
        advanced_rows.extend([full, day])

    # ใช้ objective reg:quantileerror ซึ่งเป็นชื่อ objective ของ XGBoost รุ่นปัจจุบัน
    # ต่างจาก reg:squarederror ที่ใช้กับ point forecast
    quantile_models = {}
    for quantile in (0.1, 0.9):
        quantile_estimator = _new_xgb(
            {
                **best_params,
                "objective": "reg:quantileerror",
                "quantile_alpha": quantile,
            },
            random_state=random_state,
        )
        quantile_model = make_fold_safe_preprocessing_pipeline(
            quantile_estimator, impute_feature_columns
        )
        quantile_model.fit(X_train, y_train)
        quantile_models[quantile] = quantile_model

    y_low = np.clip(
        quantile_models[0.1].predict(X_test), 0, plant_capacity_kw
    )
    y_high = np.clip(
        quantile_models[0.9].predict(X_test), 0, plant_capacity_kw
    )
    # บังคับให้ขอบล่างไม่สูงกว่าขอบบนในกรณีที่ quantile model ไขว้กันบางจุด
    y_low, y_high = np.minimum(y_low, y_high), np.maximum(y_low, y_high)
    y_test_array = np.asarray(y_test)
    picp = np.mean((y_test_array >= y_low) & (y_test_array <= y_high))
    pinaw = np.mean(y_high - y_low) / plant_capacity_kw
    interval_rows = []
    idx = pd.DatetimeIndex(test_index)
    for period, mask in [('24h', np.ones(len(idx),dtype=bool)),
                         ('06-18', (idx.hour>=6)&(idx.hour<18))]:
        interval_rows.append({'Period':period, 'N':int(mask.sum()), 'Nominal coverage':0.8,
            'PICP':float(np.mean((y_test_array[mask]>=y_low[mask]) & (y_test_array[mask]<=y_high[mask]))),
            'PINAW':float(np.mean(y_high[mask]-y_low[mask])/plant_capacity_kw)})
    interval_metrics = pd.DataFrame(interval_rows)
    interval_metrics.to_csv(
        output_dir / "table_12_prediction_interval_metrics.csv", index=False
    )
    print(f"PICP (เป้าหมายประมาณ 80%): {picp:.1%}")
    print(f"PINAW (ยิ่งเล็กยิ่งดี): {pinaw:.4f}")

    # รวม baseline กับโมเดลใหม่ แล้วคำนวณ skill score ด้วย reference เดิมชุดเดียวกัน
    advanced_results = pd.DataFrame(advanced_rows)
    results_df = pd.concat(
        [baseline_results.drop(columns=["Skill"], errors="ignore"), advanced_results],
        ignore_index=True,
    )
    results_df = _add_skill_score(results_df)
    results_df.to_csv(output_dir / "table_10_model_comparison.csv", index=False)

    return {
        "scaler": scaler,
        # คง key เดิมเป็น Ridge estimator ที่ฝึกแล้ว และเพิ่ม pipeline สำหรับ
        # การ predict บนข้อมูลดิบที่อาจมี NaN โดยไม่ทำลาย compatibility เดิม
        "ridge": model_ridge,
        "ridge_pipeline": ridge_pipeline,
        "xgb_search": search,
        "xgb": model_xgb,
        "quantile_models": quantile_models,
        "best_params": best_params,
        "y_pred_ridge": y_pred_ridge,
        "y_pred_xgb": y_pred_xgb,
        "y_low": y_low,
        "y_high": y_high,
        "results_df": results_df,
        "interval_metrics": interval_metrics,
        "ridge_top_coefficients": coef_df,
    }


def plot_selected_forecast_days(
    test_index,
    y_true,
    y_pred,
    y_low,
    y_high,
    y_persistence,
    irradiance_lag1,
    output_dir: Path,
):
    """เลือกวันฟ้าใส/ครึ้ม/ผสม และสร้างกราฟเทียบค่าจริงกับค่าพยากรณ์"""

    index = pd.DatetimeIndex(test_index)
    daily_irradiance = pd.Series(
        np.asarray(irradiance_lag1), index=index
    ).groupby(index.normalize()).mean()
    day_counts = pd.Series(index.normalize()).value_counts()
    valid_days = daily_irradiance.index.intersection(
        day_counts[day_counts >= 96].index
    )
    if len(valid_days) < 3:
        raise ValueError("ต้องมีวันทดสอบที่ข้อมูลครบอย่างน้อย 3 วันเพื่อสร้างกราฟ")

    daily_irradiance = daily_irradiance.loc[valid_days]
    clear_day = daily_irradiance.idxmax()
    cloudy_day = daily_irradiance.idxmin()
    mixed_day = (daily_irradiance - daily_irradiance.median()).abs().idxmin()
    selected = [
        (clear_day, "High prior-day irradiance"),
        (cloudy_day, "Low prior-day irradiance"),
        (mixed_day, "Median prior-day irradiance"),
    ]

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_low = np.asarray(y_low)
    y_high = np.asarray(y_high)
    y_persistence = np.asarray(y_persistence)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    for ax, (day, label) in zip(axes, selected):
        mask = index.normalize() == day
        order = np.argsort(index[mask])
        hours = np.arange(mask.sum())
        true_day = y_true[mask][order]
        pred_day = y_pred[mask][order]
        low_day = y_low[mask][order]
        high_day = y_high[mask][order]
        persist_day = y_persistence[mask][order]

        ax.plot(hours, true_day, "k-", label="Actual", linewidth=2)
        ax.plot(hours, pred_day, "b-", label="XGBoost", linewidth=1.5)
        ax.fill_between(
            hours,
            low_day,
            high_day,
            alpha=0.2,
            color="blue",
            label="10-90% band",
        )
        ax.plot(
            hours,
            persist_day,
            "--",
            color="gray",
            label="Persistence",
            linewidth=1,
        )
        ax.set_ylabel("Power (kW)")
        ax.set_title(f"{label}: {day.date()}")
        ax.set_ylim(0, 6000)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", ncol=4)

    axes[-1].set_xlabel("Interval (15-min)")
    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "figure_12_actual_vs_predicted_days.png", dpi=150)
    plt.show()
    plt.close(fig)
    return {"clear": clear_day, "cloudy": cloudy_day, "mixed": mixed_day}


def _monthly_row(period, test_features, metrics):
    expected = pd.date_range(period.start_time, period.end_time.normalize(), freq='D')
    days = summarize_forecast_day_completeness(test_features.index, test_features.P,
                                              expected_dates=expected)
    season = ('Hot' if period.month in [3,4,5] else
              'Rainy' if period.month in [6,7,8,9,10] else 'Cool')
    row = {'Month':str(period), 'Season':season, 'Test intervals':len(test_features),
           'Complete forecast days':int(days.target_complete.sum()),
           'Incomplete forecast days':int((~days.target_complete & days.valid_target_intervals.gt(0)).sum()),
           'Unscored forecast days':int(days.valid_target_intervals.eq(0).sum())}
    for name, result in metrics.items():
        for field in ['N','MAE','RMSE']:
            row[f'{name} {field}'] = result[field]
    return row


def walk_forward_baselines(df_full_raw, feature_names, build_features, output_dir,
                           min_train_months=6, capacity_kw=6000):
    rows=[]
    names={'Persistence (t-1)':'Persist(t-1)', 'Persistence (t-7)':'Persist(t-7)',
           'Historical Average':'HistAvg'}
    for period, train_raw, train_feat, test_feat in _monthly_windows(
            df_full_raw, feature_names, build_features, min_train_months):
        predictions=baseline_predictions(train_raw,test_feat,capacity_kw)
        metrics={names[name]:_evaluate(test_feat.P,pred,test_feat.index,name,capacity_kw)[0]
                 for name,pred in predictions.items()}
        rows.append(_monthly_row(period,test_feat,metrics))
    result=pd.DataFrame(rows)
    Path(output_dir).mkdir(parents=True,exist_ok=True)
    result.to_csv(Path(output_dir)/'table_09_walkforward_baselines.csv',index=False)
    return result


def walk_forward_with_models(df_full_raw, feature_names, build_features,
                             best_params=None, output_dir=None, min_train_months=6,
                             random_state=42, impute_feature_columns=None,
                             search_n_iter=30, capacity_kw=6000):
    """Tune once on the initial training months, freeze parameters, then expand.

    External hold-out best_params are deliberately rejected: their provenance
    could include observations from a monthly test window.
    """
    if best_params is not None:
        raise ValueError('Do not reuse hold-out best_params in earlier monthly windows. '
                         'Omit best_params; this function tunes on the initial period only.')
    if output_dir is None:
        raise ValueError('output_dir is required')
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    feature_names=list(feature_names)
    windows=list(_monthly_windows(df_full_raw, feature_names, build_features, min_train_months))
    if not windows:
        raise ValueError('No monthly windows remain after the initial training period')
    first_period,_,initial_train,_=windows[0]
    selection_end=initial_train.index.max()
    assert selection_end < first_period.start_time
    selection_cv=CalendarDayTimeSeriesSplit(n_splits=5,gap_days=1)
    search=_fit_xgb_search(initial_train[feature_names],initial_train.P,feature_names,
                           random_state,search_n_iter,selection_cv)
    parameters={k.removeprefix('model__'):v for k,v in search.best_params_.items()}
    selection_report={
        'strategy':'Tune once on initial training period; freeze for all monthly windows',
        'parameter_selection_start':str(initial_train.index.min()),
        'parameter_selection_end':str(selection_end),
        'first_test_start':str(first_period.start_time),
        'best_params':parameters, 'best_cv_rmse':float(-search.best_score_),
    }
    (output_dir/'table_13_monthly_parameter_provenance.json').write_text(
        json.dumps(selection_report,indent=2),encoding='utf-8')
    pd.DataFrame(search.cv_results_).to_csv(output_dir/'table_13_initial_parameter_search.csv',index=False)
    _cv_date_report(initial_train[feature_names],selection_cv).to_csv(
        output_dir/'table_13_initial_cv_date_boundaries.csv',index=False)
    print('Monthly parameter selection ends:',selection_end)
    print('First monthly test starts:',first_period.start_time)
    rows=[]
    for period,train_raw,train_feat,test_feat in windows:
        if selection_end >= period.start_time:
            raise ValueError('Monthly parameters use information after the forecast training cutoff')
        Xtr,ytr=train_feat[feature_names],train_feat.P
        Xte,yte=test_feat[feature_names],test_feat.P
        ridge=Pipeline([('imputer',SlotMedianImputer(feature_names)),
                        ('scaler',StandardScaler()),('model',Ridge(alpha=1.0))])
        ridge.fit(Xtr,ytr)
        xgb=make_fold_safe_preprocessing_pipeline(_new_xgb(parameters,random_state),feature_names)
        xgb.fit(Xtr,ytr)
        predictions=baseline_predictions(train_raw,test_feat,capacity_kw)
        predictions['Ridge']=postprocess_predictions(ridge.predict(Xte),capacity_kw)
        predictions['XGBoost']=postprocess_predictions(xgb.predict(Xte),capacity_kw)
        metrics={name:_evaluate(yte,pred,test_feat.index,name,capacity_kw)[0]
                 for name,pred in predictions.items()}
        row=_monthly_row(period,test_feat,metrics)
        row['Parameter selection end']=str(selection_end)
        row['Test start']=str(period.start_time)
        row['Parameter selection precedes test']=True
        rows.append(row)
        print(f'{period}: Ridge MAE={row["Ridge MAE"]:.2f}; XGBoost MAE={row["XGBoost MAE"]:.2f}')
    result=pd.DataFrame(rows)
    result.to_csv(output_dir/'table_13_walkforward_models.csv',index=False)
    return result


def plot_walk_forward_model_mae(wf_models, output_dir):
    if wf_models.empty:
        return
    fig,ax=plt.subplots(figsize=(14,5))
    for column,label in [('Persistence (t-1) MAE','Persistence (t-1)'),
                          ('Historical Average MAE','Historical Average'),
                          ('Ridge MAE','Ridge Regression'), ('XGBoost MAE','XGBoost')]:
        ax.plot(wf_models.Month,wf_models[column],marker='o',label=label)
    ax.set_ylabel('MAE (kW)')
    ax.set_title('Monthly evaluation with parameters selected before the first test month')
    ax.tick_params(axis='x',rotation=45)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir)/'figure_13_walkforward_model_mae.png',dpi=150)
    plt.show()
    plt.close(fig)


