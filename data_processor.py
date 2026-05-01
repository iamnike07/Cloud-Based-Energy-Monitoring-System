"""
Data Processor — Household Power Consumption
Loads, cleans, engineers features, and aggregates the raw minute-level data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ── Column names ──────────────────────────────────────────────────────────────
COLS = [
    "Date", "Time", "Global_active_power", "Global_reactive_power",
    "Voltage", "Global_intensity", "Sub_metering_1",
    "Sub_metering_2", "Sub_metering_3",
]


def load_data(filepath: str, sample_frac: float = 1.0) -> pd.DataFrame: 
    """
    Load the semicolon-delimited dataset, parse datetime, drop '?' rows,
    and optionally sample a fraction for fast prototyping.
    """
    df = pd.read_csv(
        filepath,
        sep=";",
        header=0,
        names=COLS,
        na_values=["?"],
        low_memory=False,
    )

    # Parse datetime
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
    )
    df.drop(columns=["Date", "Time"], inplace=True)
    df.set_index("Datetime", inplace=True)
    df.sort_index(inplace=True)

    # Drop missing
    df.dropna(inplace=True)

    # Cast to float
    num_cols = COLS[2:]
    df[num_cols] = df[num_cols].astype(float)

    # Optional sample for speed (stratified by date)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).sort_index()

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based and derived energy features."""
    df = df.copy()

    # Time features
    df["hour"]       = df.index.hour
    df["day_of_week"]= df.index.dayofweek          # 0=Mon
    df["month"]      = df.index.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_peak"]    = df["hour"].between(18, 22).astype(int)

    # Energy (Wh) = power (kW) * (1/60 hr) * 1000
    df["Energy_kWh"] = df["Global_active_power"] / 60.0

    # Sub-metering remainder (unaccounted appliances)
    df["Other_metering"] = (
        df["Global_active_power"] * 1000 / 60.0
        - df["Sub_metering_1"]
        - df["Sub_metering_2"]
        - df["Sub_metering_3"]
    ).clip(lower=0)

    # Power factor
    apparent = np.sqrt(
        df["Global_active_power"] ** 2 + df["Global_reactive_power"] ** 2
    )
    df["Power_factor"] = (df["Global_active_power"] / apparent.replace(0, np.nan)).fillna(1.0)

    return df


def aggregate(df: pd.DataFrame, freq: str = "H") -> pd.DataFrame:
    """
    Resample to hourly ('H'), daily ('D') or monthly ('M') totals/means.
    Returns a dataframe with energy sums and average power metrics.
    """
    agg = df.resample(freq).agg(
        Global_active_power=("Global_active_power", "mean"),
        Global_reactive_power=("Global_reactive_power", "mean"),
        Voltage=("Voltage", "mean"),
        Global_intensity=("Global_intensity", "mean"),
        Sub_metering_1=("Sub_metering_1", "sum"),
        Sub_metering_2=("Sub_metering_2", "sum"),
        Sub_metering_3=("Sub_metering_3", "sum"),
        Other_metering=("Other_metering", "sum"),
        Energy_kWh=("Energy_kWh", "sum"),
        Power_factor=("Power_factor", "mean"),
    )
    agg.dropna(inplace=True)
    return agg


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Return key KPI numbers for the dashboard header."""
    total_kwh    = df["Energy_kWh"].sum()
    avg_power    = df["Global_active_power"].mean()
    peak_power   = df["Global_active_power"].max()
    avg_voltage  = df["Voltage"].mean()
    avg_pf       = df["Power_factor"].mean()

    # Estimated cost: ₹8 per kWh (India average)
    est_cost_inr = total_kwh * 8.0

    return {
        "total_kwh":    round(total_kwh, 2),
        "avg_power_kw": round(avg_power, 3),
        "peak_power_kw":round(peak_power, 3),
        "avg_voltage_v":round(avg_voltage, 2),
        "avg_pf":       round(avg_pf, 4),
        "est_cost_inr": round(est_cost_inr, 2),
    }
