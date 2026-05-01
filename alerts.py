"""
Alerts Engine
─────────────
Evaluates rule-based thresholds on live (or latest) data and
returns structured alert objects for display in the dashboard.
"""

from dataclasses import dataclass, field
from typing import List
import pandas as pd


@dataclass
class Alert:
    level: str          # "CRITICAL" | "WARNING" | "INFO"
    category: str       # "High Usage" | "Voltage Fault" | "Anomaly" | ...
    message: str
    value: float
    threshold: float
    timestamp: str

    @property
    def icon(self) -> str:
        return {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🟢"}.get(self.level, "⚪")

    @property
    def color(self) -> str:
        return {"CRITICAL": "#FF4B4B", "WARNING": "#FFA500", "INFO": "#00CC88"}.get(
            self.level, "#AAAAAA"
        )


# ── Thresholds (configurable) ─────────────────────────────────────────────────
THRESHOLDS = {
    "critical_power_kw":  6.0,    # kW — triggers critical alert
    "warning_power_kw":   4.5,    # kW — triggers warning alert
    "low_voltage":       210.0,   # V
    "high_voltage":      250.0,   # V
    "low_power_factor":    0.85,
    "high_intensity_a":   25.0,   # A
}


def evaluate_latest(row: pd.Series) -> List[Alert]:
    """Check a single row (latest reading) against all thresholds."""
    alerts: List[Alert] = []
    ts = str(row.name) if hasattr(row, "name") else "N/A"

    power = row.get("Global_active_power", 0)
    voltage = row.get("Voltage", 230)
    intensity = row.get("Global_intensity", 0)
    pf = row.get("Power_factor", 1.0)

    # Power checks
    if power >= THRESHOLDS["critical_power_kw"]:
        alerts.append(Alert(
            level="CRITICAL", category="High Usage",
            message=f"Power consumption critically high: {power:.2f} kW",
            value=power, threshold=THRESHOLDS["critical_power_kw"], timestamp=ts,
        ))
    elif power >= THRESHOLDS["warning_power_kw"]:
        alerts.append(Alert(
            level="WARNING", category="High Usage",
            message=f"Power consumption elevated: {power:.2f} kW",
            value=power, threshold=THRESHOLDS["warning_power_kw"], timestamp=ts,
        ))
    else:
        alerts.append(Alert(
            level="INFO", category="Normal Operation",
            message=f"Power consumption normal: {power:.2f} kW",
            value=power, threshold=THRESHOLDS["warning_power_kw"], timestamp=ts,
        ))

    # Voltage checks
    if voltage < THRESHOLDS["low_voltage"]:
        alerts.append(Alert(
            level="CRITICAL", category="Voltage Fault",
            message=f"Under-voltage detected: {voltage:.1f} V",
            value=voltage, threshold=THRESHOLDS["low_voltage"], timestamp=ts,
        ))
    elif voltage > THRESHOLDS["high_voltage"]:
        alerts.append(Alert(
            level="WARNING", category="Voltage Fault",
            message=f"Over-voltage detected: {voltage:.1f} V",
            value=voltage, threshold=THRESHOLDS["high_voltage"], timestamp=ts,
        ))

    # Intensity check
    if intensity > THRESHOLDS["high_intensity_a"]:
        alerts.append(Alert(
            level="WARNING", category="High Intensity",
            message=f"High current draw: {intensity:.1f} A",
            value=intensity, threshold=THRESHOLDS["high_intensity_a"], timestamp=ts,
        ))

    # Power factor check
    if pf < THRESHOLDS["low_power_factor"]:
        alerts.append(Alert(
            level="WARNING", category="Poor Power Factor",
            message=f"Low power factor: {pf:.3f} (energy being wasted)",
            value=pf, threshold=THRESHOLDS["low_power_factor"], timestamp=ts,
        ))

    return alerts


def evaluate_bulk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scan an aggregated DataFrame and return a summary of alert events,
    useful for the historical alert log panel.
    """
    records = []
    for ts, row in df.iterrows():
        for a in evaluate_latest(row):
            if a.level in ("CRITICAL", "WARNING"):
                records.append({
                    "Timestamp":  ts,
                    "Level":      a.level,
                    "Category":   a.category,
                    "Message":    a.message,
                    "Value":      a.value,
                    "Threshold":  a.threshold,
                })
    return pd.DataFrame(records) if records else pd.DataFrame()


def optimisation_tips(df: pd.DataFrame) -> List[str]:
    """
    Rule-based optimisation suggestions based on aggregated hourly data.
    """
    tips: List[str] = []

    # Peak hour consumption
    peak   = df[df.index.hour.isin(range(18, 23))]
    offpeak = df[~df.index.hour.isin(range(18, 23))]
    if not peak.empty and not offpeak.empty:
        ratio = peak["Global_active_power"].mean() / (offpeak["Global_active_power"].mean() + 1e-9)
        if ratio > 1.4:
            tips.append("⚡ Shift heavy appliances (washing machine, dishwasher) to off-peak hours (before 6 PM or after 11 PM) to reduce peak demand charges.")

    # Sub-metering breakdown
    if "Sub_metering_1" in df.columns:
        sm1_share = df["Sub_metering_1"].sum() / (df["Energy_kWh"].sum() * 1000 / 60 + 1e-9)
        if sm1_share > 0.3:
            tips.append("🍽️ Kitchen appliances (Sub-metering 1) account for >30% of energy. Consider energy-efficient appliances or shorter usage cycles.")

    if "Sub_metering_2" in df.columns:
        sm2_share = df["Sub_metering_2"].sum() / (df["Energy_kWh"].sum() * 1000 / 60 + 1e-9)
        if sm2_share > 0.3:
            tips.append("🧺 Laundry & HVAC (Sub-metering 2) account for >30% of energy. Use cold-water wash cycles and programmable thermostats.")

    if "Sub_metering_3" in df.columns:
        sm3_share = df["Sub_metering_3"].sum() / (df["Energy_kWh"].sum() * 1000 / 60 + 1e-9)
        if sm3_share > 0.3:
            tips.append("🌡️ Electric water heater / AC (Sub-metering 3) is the top consumer. Consider a heat pump or solar water heater.")

    # Power factor
    if "Power_factor" in df.columns:
        avg_pf = df["Power_factor"].mean()
        if avg_pf < 0.90:
            tips.append(f"📉 Average power factor is {avg_pf:.2f}. Installing power factor correction capacitors can reduce reactive losses.")

    # Weekend vs weekday
    wd_avg = df[df.index.dayofweek < 5]["Global_active_power"].mean()
    we_avg = df[df.index.dayofweek >= 5]["Global_active_power"].mean()
    if we_avg > wd_avg * 1.2:
        tips.append("📅 Weekend consumption is significantly higher than weekdays. Review appliance standby modes during weekends.")

    if not tips:
        tips.append("✅ Consumption patterns look efficient. Maintain current usage habits.")

    return tips
