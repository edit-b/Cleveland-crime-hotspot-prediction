
from __future__ import annotations

import os
#Limiting CPU use 
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBClassifier

#App folder & project folder
ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent

#Declaring features
FEATURES = [
    "crime_lag_1", "crime_lag_2", "crime_lag_3", "crime_change", "crime_rolling_3",
    "hotspot_lag_1", "spatial_lag_crime", "spatial_lag_hotspot", "month_sin", "month_cos",
    "temp_mean", "temp_max", "temp_min", "rainfall_total", "rainfall_days",
    "school_holiday_days", "bank_holiday_days", "imd_decile", "income_decile",
    "employment_decile", "education_decile", "health_decile", "housing_decile",
    "living_env_decile", "la_Hartlepool", "la_Middlesbrough", "la_Redcar and Cleveland",
    "la_Stockton-on-Tees",
]
#XGBoost parameters
GRID = [
    dict(n_estimators=300, max_depth=4, learning_rate=.05, subsample=.8, colsample_bytree=.8),
    dict(n_estimators=400, max_depth=5, learning_rate=.05, subsample=.8, colsample_bytree=.8),
    dict(n_estimators=500, max_depth=6, learning_rate=.03, subsample=.7, colsample_bytree=.7),
    dict(n_estimators=300, max_depth=6, learning_rate=.05, subsample=.9, colsample_bytree=.9),
    dict(n_estimators=400, max_depth=4, learning_rate=.03, subsample=.8, colsample_bytree=.8),
    dict(n_estimators=600, max_depth=5, learning_rate=.02, subsample=.8, colsample_bytree=.8),
]

#Models
MODEL_NAMES = ["Persistence", "Logistic regression", "ARIMA", "XGBoost"]


def classifier(params, y):
    #Giving hotspot examples extra weight
    return XGBClassifier(**params, scale_pos_weight=float((y == 0).sum() / (y == 1).sum()),
                         random_state=100, eval_metric="logloss", n_jobs=1)


def scores(y, pred, probability=None):
    #Calculating the main prediction scores
    row = dict(n=len(y), accuracy=accuracy_score(y, pred),
               precision=precision_score(y, pred, zero_division=0),
               recall=recall_score(y, pred, zero_division=0),
               f1=f1_score(y, pred, zero_division=0))
    #Calculating AUC only when scores and both outcome classes are available
    row["auc"] = (roc_auc_score(y, probability)
                  if probability is not None and len(np.unique(y)) == 2 else np.nan)
    return row


def arima_area(task):
    code, values, stops = task
    rows = []
    for stop in stops:
        #Using only the history available before this forecast month
        series = np.asarray(values[:stop], dtype=float)
        status = "converged"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ARIMA(series, order=(1, 0, 1)).fit()
                forecast = float(fit.forecast(steps=1)[0])
            #Flagging fits not fully converged during training
            if not fit.mle_retvals.get("converged", True):
                status = "non_converged"
            if not np.isfinite(forecast):
                raise ValueError("Non-finite forecast")
            #Preventing negative incident counts
            forecast = max(0., forecast)
        #Using the historical average if fitting fails or the forecast is invalid
        except Exception as exc:
            forecast = max(0., float(np.mean(series)))
            status = "mean_fallback:" + type(exc).__name__
        rows.append((code, stop, forecast, status))
    return rows


def main(workers):
    #Saving generated files 
    out = ROOT / "data"
    out.mkdir(parents=True, exist_ok=True)
    #Loading the original project data
    source = PROJECT_ROOT / "Data/crime_features_model.csv"
    df = pd.read_csv(source).sort_values(["LSOA code", "Month"]).reset_index(drop=True)
    #Checking for duplicate rows, missing features, unexpected area counts
    assert not df.duplicated(["LSOA code", "Month"]).any()
    assert not df[FEATURES].isna().any().any()
    assert df.groupby("Month").size().eq(359).all()
    assert df.groupby("Month").hotspot.sum().eq(36).all()
    #Ranking areas by crime count 
    df["rank_now"] = df.groupby("Month").crime_count.rank(ascending=False, method="first").astype(int)
    assert df.hotspot.eq(df.rank_now.le(36).astype(int)).all()
    #Attaching next month’s actual outcome to each area’s current-month row
    groups = df.groupby("LSOA code")
    df["target_month"] = groups.Month.shift(-1)
    df["actual_hotspot"] = groups.hotspot.shift(-1)
    df["actual_crime_count"] = groups.crime_count.shift(-1)
    complete = df.copy()
    #Removing final-month rows (next-month outcomes are unavailable)
    df = df.dropna(subset=["actual_hotspot"]).copy()
    df["actual_hotspot"] = df.actual_hotspot.astype(int)
    expected = (pd.PeriodIndex(df.Month, freq="M") + 1).astype(str)
    assert np.array_equal(df.target_month, expected)
    #Reserving the last 8 months for testing
    months = sorted(df.Month.unique())
    train = df[df.Month.isin(months[:-8])].copy()
    test = df[df.Month.isin(months[-8:])].copy().reset_index(drop=True)

    # Using the last 6 training months to choose XGBoost settings
    val_months = sorted(train.Month.unique())[-6:]
    inner = train[~train.Month.isin(val_months)]
    val = train[train.Month.isin(val_months)]
    X, y = train[FEATURES], train.actual_hotspot
    Xt, yt = test[FEATURES], test.actual_hotspot

    #Scaling the inputs and training LR
    scaler = StandardScaler()
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=100)
    lr.fit(scaler.fit_transform(X), y)
    lr_proba = lr.predict_proba(scaler.transform(Xt))[:, 1]
    lr_pred = lr.predict(scaler.transform(Xt))

    #Testing 6 XGBoost settings and different hotspot thresholds
    #Original and revised selection methods are separate
    legacy_best = (-1., None)
    revised_best = (-1., None, None)
    tuning_rows = []
    for number, params in enumerate(GRID):
        model = classifier(params, inner.actual_hotspot)
        model.fit(inner[FEATURES], inner.actual_hotspot)
        probs = model.predict_proba(val[FEATURES])[:, 1]
        for threshold in np.arange(.3, .75, .05):
            score = f1_score(val.actual_hotspot, probs >= threshold, zero_division=0)
            if score > legacy_best[0]: legacy_best = (float(score), params)
        for threshold in np.arange(.3, .75, .025):
            score = f1_score(val.actual_hotspot, probs >= threshold, zero_division=0)
            tuning_rows.append(dict(configuration=number, threshold=float(threshold), f1=float(score), **params))
            if score > revised_best[0]: revised_best = (float(score), params, float(threshold))
        print(f"Validated XGBoost configuration {number + 1}/6", flush=True)
    #Recreating the original method (selected its threshold after refitting)
    legacy_xgb = classifier(legacy_best[1], y)
    legacy_xgb.fit(X, y)
    legacy_val_probs = legacy_xgb.predict_proba(val[FEATURES])[:, 1]
    legacy_thr, legacy_f1 = .5, -1.
    for threshold in np.arange(.3, .75, .025):
        score = f1_score(val.actual_hotspot, legacy_val_probs >= threshold, zero_division=0)
        if score > legacy_f1: legacy_f1, legacy_thr = float(score), float(threshold)
    legacy_probs = legacy_xgb.predict_proba(Xt)[:, 1]

    #Training the revised model using settings and the threshold chosen earlier
    revised_xgb = classifier(revised_best[1], y)
    revised_xgb.fit(X, y)
    revised_probs = revised_xgb.predict_proba(Xt)[:, 1]

    #Arranging each area’s monthly counts for ARIMA forecasting
    wide = complete.pivot(index="Month", columns="LSOA code", values="crime_count").sort_index()
    first = wide.index.get_loc(months[-8])
    stops = list(range(first, first + 9))
    tasks = [(code, wide[code].tolist(), stops) for code in wide.columns]
    arima_rows = []
    #Forecasting several areas at the same time
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, rows in enumerate(pool.map(arima_area, tasks), 1):
            arima_rows.extend(rows)
            if i % 40 == 0 or i == len(tasks):
                print(f"ARIMA: {i}/{len(tasks)} areas completed", flush=True)
    #Recording the forecast dates and fitting status
    ar = pd.DataFrame(arima_rows, columns=["lsoa_code", "stop", "forecast", "fit_status"])
    ar["forecast_month"] = ar.stop.map(lambda stop: wide.index[stop])
    ar["history_end_month"] = ar.stop.map(lambda stop: wide.index[stop - 1])
    ar["pred"] = 0
    for stop, sub in ar.groupby("stop"):
        #Select the 36 areas with the highest forecast counts
        series = sub.set_index("lsoa_code").forecast.sort_values(ascending=False)
        chosen = set(series.head(36).index)
        ar.loc[sub.index, "pred"] = sub.lsoa_code.isin(chosen).astype(int)
    ar.to_csv(out / "arima_diagnostics.csv", index=False)

    #Preparing column names for the app
    base = test.rename(columns={"LSOA code": "lsoa_code", "LSOA name": "lsoa_name",
                               "Local_Authority": "local_authority", "Month": "observation_month",
                               "hotspot": "current_hotspot", "crime_count": "current_crime_count"})
    #Marking areas that changed status / were near the hotspot cutoff
    base["is_transition"] = base.current_hotspot.ne(base.actual_hotspot)
    base["boundary_zone"] = base.rank_now.between(26, 46)
    base["actual_crime_count"] = base.actual_crime_count.astype(int)
    keep = ["lsoa_code", "lsoa_name", "local_authority", "observation_month", "target_month",
            "current_crime_count", "actual_crime_count", "current_hotspot", "actual_hotspot",
            "rank_now", "boundary_zone", "is_transition", "imd_decile"]
    exports = []

    #Building separate prediction tables for the original and revised setups
    for run in ["submitted_reproduction", "portfolio_v1"]:
        #Preserving the original date mismatch only in the reproduction table
        key = "observation_month" if run == "submitted_reproduction" else "target_month"
        joined = base.merge(ar, left_on=["lsoa_code", key], right_on=["lsoa_code", "forecast_month"],
                            how="left", validate="one_to_one")
        assert joined.pred.notna().all()
        assert joined.lsoa_code.equals(base.lsoa_code)
        probabilities = legacy_probs if run == "submitted_reproduction" else revised_probs
        threshold = legacy_thr if run == "submitted_reproduction" else revised_best[2]
        predictions = [base.current_hotspot.to_numpy(), lr_pred, joined.pred.to_numpy(),
                       (probabilities >= threshold).astype(int)]
        for name, pred in zip(MODEL_NAMES, predictions):
            result = base[keep].copy()
            result["run"] = run
            result["model"] = name
            result["predicted_hotspot"] = pred
            result["probability"] = lr_proba if name == "Logistic regression" else probabilities if name == "XGBoost" else np.nan
            result["decision_threshold"] = .5 if name == "Logistic regression" else threshold if name == "XGBoost" else np.nan
            result["predicted_crime_count"] = joined.forecast if name == "ARIMA" else np.nan
            result["forecast_for_month"] = joined.forecast_month if name == "ARIMA" else result.target_month
            result["history_end_month"] = joined.history_end_month if name == "ARIMA" else ""
            result["correct"] = result.predicted_hotspot.eq(result.actual_hotspot)
            #Labelling each prediction for summary 
            result["outcome"] = np.select([
                result.actual_hotspot.eq(1) & result.predicted_hotspot.eq(1),
                result.actual_hotspot.eq(1) & result.predicted_hotspot.eq(0),
                result.actual_hotspot.eq(0) & result.predicted_hotspot.eq(1)],
                ["Correct hotspot", "Missed hotspot", "False alarm"], default="Correct non-hotspot")
            exports.append(result)

    #Saving individual area predictions for both versions
    all_predictions = pd.concat(exports, ignore_index=True)
    for run, sub in all_predictions.groupby("run"):
        sub.to_csv(out / f"predictions_{run}.csv", index=False)

    rows = []

    #Comparing models across all areas and evaluation groups
    for (run, model), sub in all_predictions.groupby(["run", "model"]):
        scopes = {"All areas": sub, "Near cutoff (ranks 26–46)": sub[sub.boundary_zone],
                  "Outside cutoff zone": sub[~sub.boundary_zone], "Status changes": sub[sub.is_transition]}
        for scope, selected in scopes.items():
            prob = selected.probability if selected.probability.notna().all() else None
            rows.append(dict(run=run, model=model, scope=scope,
                             **scores(selected.actual_hotspot, selected.predicted_hotspot, prob)))
    
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(out / "validation_search.csv", index=False)

    #Comparing the rerun with the saved research results
    original = pd.read_csv(PROJECT_ROOT / "Model_outputs/model_results.csv", index_col=0)
    original.index = MODEL_NAMES
    reproduced = metrics.query("run == 'submitted_reproduction' and scope == 'All areas'").set_index("model")
    checks = []
    for model in MODEL_NAMES:
        for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
            expected_value, actual_value = original.loc[model, metric], reproduced.loc[model, metric]
            if pd.isna(expected_value): continue
            checks.append(dict(model=model, metric=metric, submitted=float(expected_value),
                               reproduced=float(actual_value), delta=float(actual_value - expected_value),
                               matches=bool(np.isclose(actual_value, expected_value, atol=1e-10, rtol=0))))
    pd.DataFrame(checks).to_csv(out / "reproduction_check.csv", index=False)

    #Saving dates, model settings, software versions
    metadata = dict(
        run="portfolio_v1", seed=100, n_areas=359, hotspot_count=36,
        train_observation_months=[months[0], months[-9]],
        train_target_months=[train.target_month.min(), train.target_month.max()],
        test_observation_months=[months[-8], months[-1]],
        test_target_months=[test.target_month.min(), test.target_month.max()],
        validation_observation_months=[val_months[0], val_months[-1]],
        inner_train_target_end=inner.target_month.max(),
        validation_target_start=val.target_month.min(),
        xgboost_parameters=revised_best[1], xgboost_threshold=revised_best[2],
        validation_f1=revised_best[0],
        reproduction_xgboost_parameters=legacy_best[1], reproduction_xgboost_threshold=legacy_thr,
        arima_fit_status_counts=ar.fit_status.value_counts().to_dict(),
        submitted_metric_matches=all(c["matches"] for c in checks),
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        python=platform.python_version(),
        packages={p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "statsmodels"]},
        
    )
    
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    print(metrics.query("scope == 'All areas'")[["run", "model", "accuracy", "f1"]].to_string(index=False), flush=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1: parser.error("workers must be positive")
    main(args.workers)
