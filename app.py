"""
Cloud-Based Energy Consumption Monitoring & Optimization System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Streamlit Dashboard  |  Author: Nikesh Sinha (RA2311027010171)
SRM Institute of Science and Technology
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys

# ── Make local modules importable ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from data_processor import load_data, engineer_features, aggregate, get_summary_stats
from ml_model import (
    train_forecast_model, forecast_next_24h,
    detect_anomalies, get_anomaly_summary,
)
from alerts import evaluate_latest, evaluate_bulk, optimisation_tips, THRESHOLDS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Monitoring System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
  .alert-box {
    padding: 10px 14px; border-radius: 8px;
    margin-bottom: 8px; font-size: 0.9rem;
  }
  .tip-box {
    padding: 10px 14px; border-radius: 8px;
    background: #1a2a1a; border-left: 4px solid #00CC88;
    margin-bottom: 8px; color: #e0ffe0;
  }
  h1 { color: #00C49F; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Data & Model Controls
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/1x1.png",
        width=1,
    )
    st.title("⚡ Energy Monitor")
    st.markdown("---")

    st.subheader("📂 Data Source")
    uploaded = st.file_uploader(
        "Upload dataset (.txt / .csv)",
        type=["txt", "csv"],
        help="Use the household_power_consumption.txt dataset",
    )
    default_path = st.text_input(
        "Or enter file path",
        value="household_power_consumption.txt",
    )

    st.subheader("⚙️ Settings")
    sample_pct = st.slider("Data sample %", 5, 100, 20, step=5,
                           help="Lower = faster loading, higher = more accurate")
    agg_freq   = st.selectbox("Aggregation frequency", ["h", "D", "W"], index=0,
                              help="H=Hourly, D=Daily, W=Weekly")
    date_range = st.date_input("Date range filter", [])

    st.markdown("---")
    st.subheader("🚨 Alert Thresholds")
    THRESHOLDS["warning_power_kw"]  = st.number_input("Warning power (kW)",  value=4.5, step=0.5)
    THRESHOLDS["critical_power_kw"] = st.number_input("Critical power (kW)", value=6.0, step=0.5)

    st.markdown("---")
    st.caption("👨‍🎓 Nikesh Sinha · RA2311027010171\n\n🏫 SRM Institute of Science & Technology")

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading & preprocessing data…", ttl=3600)
def get_data(path_or_bytes, sample_frac):
    if isinstance(path_or_bytes, bytes):
        import io, tempfile, os
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        tmp.write(path_or_bytes)
        tmp.close()
        df = load_data(tmp.name, sample_frac)
        os.unlink(tmp.name)
    else:
        df = load_data(path_or_bytes, sample_frac)
    return engineer_features(df)


try:
    src = uploaded.read() if uploaded else default_path
    df_raw = get_data(src, sample_pct / 100)
except FileNotFoundError:
    st.error(
        "⚠️ Dataset not found. Please upload the file using the sidebar "
        "or enter the correct file path."
    )
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Apply date filter
if date_range and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    df_raw = df_raw.loc[start:end]
    if df_raw.empty:
        st.warning("No data in selected date range.")
        st.stop()

# Aggregated view
df_agg = aggregate(df_raw, agg_freq)
stats  = get_summary_stats(df_raw)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("☁️ Cloud-Based Energy Consumption Monitoring & Optimization")
st.markdown(
    "*Leveraging IoT + ML for Sustainable Energy (SDG 7)*  "
    f"| **{len(df_raw):,}** readings loaded"
    f" | {df_raw.index.min().date()} → {df_raw.index.max().date()}"
)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("🔌 Total Energy", f"{stats['total_kwh']:,.1f} kWh")
k2.metric("⚡ Avg Power",    f"{stats['avg_power_kw']} kW")
k3.metric("🔥 Peak Power",   f"{stats['peak_power_kw']} kW")
k4.metric("🔋 Avg Voltage",  f"{stats['avg_voltage_v']} V")
k5.metric("📐 Power Factor", f"{stats['avg_pf']:.3f}")
k6.metric("💰 Est. Cost",    f"₹{stats['est_cost_inr']:,.0f}")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Real-Time Dashboard",
    "🤖 AI Forecasting",
    "🚨 Anomaly Detection",
    "🔔 Alerts & Optimization",
    "📈 Deep Analytics",
])


# ════════════ TAB 1 — REAL-TIME DASHBOARD ════════════════════════════════════
with tab1:
    st.subheader("Global Active Power Over Time")

    # Downsample for plotting speed
    plot_df = df_agg.reset_index()

    fig_power = px.area(
        plot_df, x="Datetime", y="Global_active_power",
        labels={"Global_active_power": "Active Power (kW)", "Datetime": ""},
        color_discrete_sequence=["#00C49F"],
        template="plotly_dark",
    )
    fig_power.add_hline(y=THRESHOLDS["warning_power_kw"],  line_dash="dash",
                        line_color="orange", annotation_text="Warning")
    fig_power.add_hline(y=THRESHOLDS["critical_power_kw"], line_dash="dash",
                        line_color="red",    annotation_text="Critical")
    fig_power.update_layout(height=320, margin=dict(t=20, b=20))
    st.plotly_chart(fig_power, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Sub-Metering Breakdown")
        sm_cols = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "Other_metering"]
        sm_avail = [c for c in sm_cols if c in df_agg.columns]
        sm_labels = {
            "Sub_metering_1":  "Kitchen",
            "Sub_metering_2":  "Laundry / HVAC",
            "Sub_metering_3":  "Water Heater / AC",
            "Other_metering":  "Other Appliances",
        }
        sm_vals = {sm_labels[c]: df_agg[c].sum() for c in sm_avail}
        fig_pie = px.pie(
            names=list(sm_vals.keys()),
            values=list(sm_vals.values()),
            color_discrete_sequence=px.colors.qualitative.Set2,
            template="plotly_dark",
            hole=0.4,
        )
        fig_pie.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.subheader("Voltage Distribution")
        fig_volt = px.histogram(
            df_raw, x="Voltage", nbins=60,
            color_discrete_sequence=["#635DFF"],
            template="plotly_dark",
            labels={"Voltage": "Voltage (V)", "count": "Frequency"},
        )
        fig_volt.add_vline(x=230, line_dash="dash", line_color="green",
                           annotation_text="Nominal 230V")
        fig_volt.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig_volt, use_container_width=True)

    st.subheader("Average Power by Hour of Day")
    hourly_avg = df_raw.groupby("hour")["Global_active_power"].mean().reset_index()
    fig_hour = px.bar(
        hourly_avg, x="hour", y="Global_active_power",
        labels={"hour": "Hour of Day", "Global_active_power": "Avg Power (kW)"},
        color="Global_active_power",
        color_continuous_scale="Teal",
        template="plotly_dark",
    )
    fig_hour.update_layout(height=300, margin=dict(t=10, b=20), coloraxis_showscale=False)
    st.plotly_chart(fig_hour, use_container_width=True)


# ════════════ TAB 2 — AI FORECASTING ═════════════════════════════════════════
with tab2:
    st.subheader("🤖 AI-Based Demand Forecasting (Random Forest)")
    st.info(
        "Model is trained on 80% historical hourly data and tested on the remaining 20%. "
        "Features: 48-hour lags, rolling statistics, time-of-day, weekday/weekend indicators."
    )

    @st.cache_resource(show_spinner="Training forecasting model…")
    def cached_train(agg_key):
        hourly = aggregate(df_raw, "h")
        return train_forecast_model(hourly)

    with st.spinner("Training model (cached after first run)…"):
        model, scaler, metrics, feat_imp = cached_train(str(df_raw.index[0]))

    # Metrics
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("MAE",  f"{metrics['MAE']} kW")
    mc2.metric("RMSE", f"{metrics['RMSE']} kW")
    mc3.metric("R²",   f"{metrics['R²']}")

    # Actual vs Predicted
    st.subheader("Actual vs Predicted — Test Set")
    actual_pred_df = pd.DataFrame({
        "Datetime": model._test_index,
        "Actual":   model._y_test,
        "Predicted":model._y_pred,
    })
    fig_ap = go.Figure()
    fig_ap.add_trace(go.Scatter(
        x=actual_pred_df["Datetime"], y=actual_pred_df["Actual"],
        name="Actual", line=dict(color="#00C49F", width=1.5)))
    fig_ap.add_trace(go.Scatter(
        x=actual_pred_df["Datetime"], y=actual_pred_df["Predicted"],
        name="Predicted", line=dict(color="#FF6B6B", width=1.5, dash="dot")))
    fig_ap.update_layout(
        template="plotly_dark", height=320,
        legend=dict(orientation="h", y=1.02),
        margin=dict(t=20, b=20),
        yaxis_title="Active Power (kW)",
    )
    st.plotly_chart(fig_ap, use_container_width=True)

    # 24h forecast
    st.subheader("Next 24-Hour Demand Forecast")
    hourly_full = aggregate(df_raw, "h")
    fc_df = forecast_next_24h(model, scaler, hourly_full)
    fc_df = fc_df.reset_index()

    fig_fc = px.line(
        fc_df, x="Datetime", y="Forecast_kW",
        labels={"Forecast_kW": "Forecasted Power (kW)", "Datetime": ""},
        color_discrete_sequence=["#FFD700"],
        template="plotly_dark",
        markers=True,
    )
    fig_fc.update_layout(height=300, margin=dict(t=10, b=20))
    st.plotly_chart(fig_fc, use_container_width=True)
    st.dataframe(fc_df.set_index("Datetime").round(4), use_container_width=True, height=200)

    # Feature importance
    st.subheader("Top 10 Feature Importances")
    fig_fi = px.bar(
        feat_imp, x="Importance", y="Feature",
        orientation="h", color="Importance",
        color_continuous_scale="Teal", template="plotly_dark",
    )
    fig_fi.update_layout(height=340, margin=dict(t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig_fi, use_container_width=True)


# ════════════ TAB 3 — ANOMALY DETECTION ══════════════════════════════════════
with tab3:
    st.subheader("🚨 Anomaly Detection — Isolation Forest")
    contamination = st.slider("Expected anomaly rate (%)", 1, 10, 2) / 100

    @st.cache_data(show_spinner="Detecting anomalies…")
    def cached_anomalies(cont, idx_key):
        sample = df_raw.sample(min(50_000, len(df_raw)), random_state=42).sort_index()
        return detect_anomalies(sample, cont)

    df_anom = cached_anomalies(contamination, str(df_raw.index[0]))
    summary = get_anomaly_summary(df_anom)

    ac1, ac2, ac3 = st.columns(3)
    ac1.metric("Total Readings", f"{summary['total']:,}")
    ac2.metric("Anomalies Found", f"{summary['n_anom']:,}")
    ac3.metric("Anomaly Rate", f"{summary['pct']} %")

    # Scatter plot
    normal = df_anom[df_anom["anomaly"] == 1]
    anom   = df_anom[df_anom["anomaly"] == -1]

    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scattergl(
        x=normal.index, y=normal["Global_active_power"],
        mode="markers", name="Normal",
        marker=dict(color="#00C49F", size=2, opacity=0.4),
    ))
    fig_sc.add_trace(go.Scattergl(
        x=anom.index, y=anom["Global_active_power"],
        mode="markers", name="Anomaly",
        marker=dict(color="#FF4B4B", size=5, symbol="x"),
    ))
    fig_sc.update_layout(
        template="plotly_dark", height=360,
        yaxis_title="Active Power (kW)",
        xaxis_title="",
        margin=dict(t=20, b=20),
        legend=dict(orientation="h", y=1.02),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # Anomaly score distribution
    st.subheader("Anomaly Score Distribution")
    fig_hist = px.histogram(
        df_anom, x="anomaly_score", nbins=80,
        color="anomaly", barmode="overlay",
        color_discrete_map={1: "#00C49F", -1: "#FF4B4B"},
        labels={"anomaly_score": "Isolation Score (lower = more anomalous)",
                "anomaly": "Label"},
        template="plotly_dark",
    )
    fig_hist.update_layout(height=280, margin=dict(t=10, b=20))
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Top 5 Most Anomalous Readings")
    st.dataframe(summary["worst"], use_container_width=True)


# ════════════ TAB 4 — ALERTS & OPTIMISATION ══════════════════════════════════
with tab4:
    st.subheader("🔔 Live Alerts — Latest Reading")
    latest = df_raw.iloc[-1]
    current_alerts = evaluate_latest(latest)

    for alert in current_alerts:
        st.markdown(
            f'<div class="alert-box" style="background:{alert.color}22;border-left:4px solid {alert.color}">'
            f'{alert.icon} <strong>[{alert.level}] {alert.category}</strong><br>'
            f'{alert.message} &nbsp;|&nbsp; <em>{alert.timestamp}</em>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("📋 Historical Alert Log")

    with st.spinner("Scanning for historical alerts…"):
        hist_sample = df_agg.sample(min(5000, len(df_agg)), random_state=0).sort_index()
        alert_log   = evaluate_bulk(hist_sample)

    if not alert_log.empty:
        level_filter = st.multiselect(
            "Filter by level", ["CRITICAL", "WARNING"],
            default=["CRITICAL", "WARNING"],
        )
        filtered_log = alert_log[alert_log["Level"].isin(level_filter)]
        st.dataframe(filtered_log.head(200), use_container_width=True, height=300)

        fig_alog = px.histogram(
            alert_log, x="Timestamp", color="Level",
            barmode="stack",
            color_discrete_map={"CRITICAL": "#FF4B4B", "WARNING": "#FFA500"},
            template="plotly_dark",
            labels={"Timestamp": "", "count": "# Alerts"},
        )
        fig_alog.update_layout(height=250, margin=dict(t=10, b=20))
        st.plotly_chart(fig_alog, use_container_width=True)
    else:
        st.success("No historical alerts found in sampled data.")

    st.divider()
    st.subheader("💡 AI-Powered Optimisation Tips")
    for tip in optimisation_tips(df_agg):
        st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)


# ════════════ TAB 5 — DEEP ANALYTICS ═════════════════════════════════════════
with tab5:
    st.subheader("🗓️ Daily Energy Heatmap")

    daily = df_raw.copy()
    daily["date"] = daily.index.date
    daily["hour"] = daily.index.hour
    hm_df = daily.groupby(["date", "hour"])["Global_active_power"].mean().reset_index()
    hm_pivot = hm_df.pivot(index="date", columns="hour", values="Global_active_power")

    fig_hm = px.imshow(
        hm_pivot,
        labels=dict(x="Hour of Day", y="Date", color="Power (kW)"),
        color_continuous_scale="Teal",
        template="plotly_dark",
        aspect="auto",
    )
    fig_hm.update_layout(height=400, margin=dict(t=20, b=20))
    st.plotly_chart(fig_hm, use_container_width=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Monthly Energy Consumption")
        monthly = df_raw.resample("ME")["Energy_kWh"].sum().reset_index()
        monthly.columns = ["Month", "Energy_kWh"]
        fig_mon = px.bar(
            monthly, x="Month", y="Energy_kWh",
            color="Energy_kWh", color_continuous_scale="Teal",
            template="plotly_dark",
            labels={"Energy_kWh": "Energy (kWh)", "Month": ""},
        )
        fig_mon.update_layout(height=300, margin=dict(t=10, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig_mon, use_container_width=True)

    with col_r:
        st.subheader("Weekday vs Weekend Power Profile")
        df_raw["day_type"] = df_raw["is_weekend"].map({0: "Weekday", 1: "Weekend"})
        wdwe = df_raw.groupby(["hour", "day_type"])["Global_active_power"].mean().reset_index()
        fig_wdwe = px.line(
            wdwe, x="hour", y="Global_active_power",
            color="day_type",
            color_discrete_map={"Weekday": "#00C49F", "Weekend": "#FFD700"},
            template="plotly_dark",
            labels={"hour": "Hour of Day", "Global_active_power": "Avg Power (kW)",
                    "day_type": "Day Type"},
            markers=True,
        )
        fig_wdwe.update_layout(height=300, margin=dict(t=10, b=20))
        st.plotly_chart(fig_wdwe, use_container_width=True)

    st.subheader("Correlation Matrix — All Features")
    num_cols = [
        "Global_active_power", "Global_reactive_power", "Voltage",
        "Global_intensity", "Sub_metering_1", "Sub_metering_2",
        "Sub_metering_3", "Power_factor",
    ]
    corr_cols = [c for c in num_cols if c in df_raw.columns]
    corr_samp = df_raw[corr_cols].sample(min(20_000, len(df_raw)), random_state=42)
    corr = corr_samp.corr().round(2)

    fig_corr = px.imshow(
        corr, text_auto=True,
        color_continuous_scale="RdBu", zmin=-1, zmax=1,
        template="plotly_dark",
    )
    fig_corr.update_layout(height=450, margin=dict(t=20, b=20))
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Power Factor Over Time")
    pf_df = df_agg[["Power_factor"]].reset_index()
    fig_pf = px.line(
        pf_df, x="Datetime", y="Power_factor",
        color_discrete_sequence=["#A78BFA"],
        template="plotly_dark",
        labels={"Power_factor": "Power Factor", "Datetime": ""},
    )
    fig_pf.add_hline(y=0.90, line_dash="dash", line_color="orange",
                     annotation_text="0.90 target")
    fig_pf.update_layout(height=280, margin=dict(t=10, b=20))
    st.plotly_chart(fig_pf, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center>☁️ <b>Smart Tech for a Sustainable Future</b> · "
    "Efficient today, sustainable tomorrow · "
    "SRM Institute of Science and Technology</center>",
    unsafe_allow_html=True,
)
