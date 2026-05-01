"""
ML Models
─────────
1. Demand Forecasting  — Random Forest Regressor (next 24 h from hourly data)
2. Anomaly Detection   — Isolation Forest on minute-level readings
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


# ── 1. DEMAND FORECASTING ─────────────────────────────────────────────────────

def make_forecast_features(hourly_df: pd.DataFrame, target_col: str = "Global_active_power") -> pd.DataFrame:
    """
    Create lag + rolling features for supervised learning on hourly data.
    Returns a dataframe with features and the target column.
    """
    df = hourly_df[[target_col]].copy()

    lags = [1, 2, 3, 6, 12, 24, 48]
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)

    windows = [3, 6, 12, 24]
    for w in windows:
        df[f"roll_mean_{w}"] = df[target_col].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"]  = df[target_col].shift(1).rolling(w).std()

    df["hour"]        = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"]       = df.index.month
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_peak"]     = df["hour"].between(18, 22).astype(int)

    df.dropna(inplace=True)
    return df


def train_forecast_model(hourly_df: pd.DataFrame, target_col: str = "Global_active_power"):
    """
    Train a Random Forest forecasting model.
    Returns (model, scaler, metrics_dict, feature_importance_df).
    """
    feat_df = make_forecast_features(hourly_df, target_col)

    X = feat_df.drop(columns=[target_col])
    y = feat_df[target_col]

    # 80/20 temporal split
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)

    metrics = {
        "MAE":  round(mean_absolute_error(y_test, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "R²":   round(r2_score(y_test, y_pred), 4),
    }

    fi = pd.DataFrame({
        "Feature":    X.columns,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(10)

    # Attach test predictions for display
    model._test_index = y_test.index
    model._y_test     = y_test.values
    model._y_pred     = y_pred

    return model, scaler, metrics, fi


def forecast_next_24h(model, scaler, hourly_df: pd.DataFrame,
                      target_col: str = "Global_active_power") -> pd.DataFrame:
    """
    Generate 24-hour ahead predictions by iterative feature reconstruction.
    Returns a DataFrame with 'Datetime' and 'Forecast_kW' columns.
    """
    feat_df = make_forecast_features(hourly_df, target_col)
    last_features = feat_df.drop(columns=[target_col]).iloc[[-1]]

    forecasts = []
    last_ts   = hourly_df.index[-1]

    history = list(hourly_df[target_col].values)

    for h in range(1, 25):
        ts = last_ts + pd.Timedelta(hours=h)

        # Rebuild lag / rolling features from running history
        row = {}
        lags    = [1, 2, 3, 6, 12, 24, 48]
        windows = [3, 6, 12, 24]
        for lag in lags:
            row[f"lag_{lag}"] = history[-lag] if len(history) >= lag else np.nan
        for w in windows:
            slice_ = history[-w:] if len(history) >= w else history
            row[f"roll_mean_{w}"] = np.mean(slice_)
            row[f"roll_std_{w}"]  = np.std(slice_) if len(slice_) > 1 else 0.0
        row["hour"]        = ts.hour
        row["day_of_week"] = ts.dayofweek
        row["month"]       = ts.month
        row["is_weekend"]  = int(ts.dayofweek in [5, 6])
        row["is_peak"]     = int(18 <= ts.hour <= 22)

        X_row   = pd.DataFrame([row])[last_features.columns]
        X_row_s = scaler.transform(X_row)
        pred    = float(model.predict(X_row_s)[0])

        forecasts.append({"Datetime": ts, "Forecast_kW": round(max(pred, 0), 4)})
        history.append(pred)

    return pd.DataFrame(forecasts).set_index("Datetime")


# ── 2. ANOMALY DETECTION ──────────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame, contamination: float = 0.02) -> pd.DataFrame:
    """
    Run Isolation Forest on power consumption features.
    Adds 'anomaly' column: 1 = normal, -1 = anomaly.
    Returns the dataframe with anomaly labels.
    """
    feature_cols = [
        "Global_active_power", "Global_reactive_power",
        "Voltage", "Global_intensity",
        "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    ]
    # Use available cols
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    labels = iso.fit_predict(X_s)

    result = df.copy()
    result["anomaly"]       = labels          # -1 = anomaly
    result["anomaly_score"] = iso.score_samples(X_s)  # lower = more anomalous
    return result


def get_anomaly_summary(df_with_anomalies: pd.DataFrame) -> dict:
    """Return counts and worst anomalies."""
    total    = len(df_with_anomalies)
    n_anom   = (df_with_anomalies["anomaly"] == -1).sum()
    worst    = df_with_anomalies[df_with_anomalies["anomaly"] == -1].nsmallest(
                   5, "anomaly_score"
               )[["Global_active_power", "Voltage", "anomaly_score"]]
    return {
        "total":   total,
        "n_anom":  int(n_anom),
        "pct":     round(n_anom / total * 100, 2),
        "worst":   worst,
    }
