# ☁️ Cloud-Based Energy Consumption Monitoring & Optimization System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=for-the-badge&logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75?style=for-the-badge&logo=plotly)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![SDG](https://img.shields.io/badge/UN%20SDG-7%20Clean%20Energy-FCC30B?style=for-the-badge)

<br/>

> **Leveraging Cloud Technology, IoT & Machine Learning for Sustainable Energy**

*A real-time energy monitoring and AI-powered optimization dashboard built on the UCI Household Power Consumption dataset.*

<br/>

| 👨‍🎓 Author | 🆔 Roll Number | 🏫 Institution |
|---|---|---|
| Nikesh Sinha | RA2311027010171 | SRM Institute of Science and Technology |

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Architecture](#-architecture)
- [ML Models](#-ml-models)
- [Alerts & Thresholds](#-alerts--thresholds)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dashboard Tabs](#-dashboard-tabs)
- [SDG Alignment](#-sdg-alignment)
- [Tech Stack](#-tech-stack)
- [Cloud Deployment](#-cloud-deployment)
- [License](#-license)

---

## 🌟 Overview

Global energy demand is increasing rapidly, leading to higher costs and environmental impact. Traditional monitoring systems offer **no real-time visibility**, leading to:

- ❌ Energy wastage from unmonitored appliances
- ❌ Peak load issues causing outages
- ❌ No data-driven insights for optimization
- ❌ Poor integration with renewable energy sources

This project addresses all of the above with a **full-stack data science solution** — a Streamlit dashboard powered by machine learning models that provides real-time monitoring, demand forecasting, anomaly detection, and intelligent optimization recommendations.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 **Real-Time Dashboard** | Live charts for power, voltage, sub-metering and hourly usage profiles |
| 🤖 **AI Demand Forecasting** | Random Forest model predicting next 24 hours of energy demand |
| 🚨 **Anomaly Detection** | Isolation Forest flagging abnormal consumption events automatically |
| 🔔 **Automated Alerts** | Rule-based CRITICAL / WARNING / INFO alerts for power, voltage and current |
| 💡 **Optimization Tips** | Pattern-based AI recommendations for reducing energy waste |
| 📈 **Deep Analytics** | Daily heatmaps, monthly trends, weekday vs weekend profiles, correlation matrix |
| ⚙️ **Configurable** | All thresholds, date ranges, and sampling rates adjustable from the sidebar |

---

## 🗂️ Project Structure

```
energy_monitoring/
│
├── app.py                  ←  Streamlit dashboard (main entry point)
├── data_processor.py       ←  Data loading, cleaning, feature engineering, aggregation
├── ml_model.py             ←  Random Forest forecasting + Isolation Forest anomaly detection
├── alerts.py               ←  Rule-based alert engine + AI optimization tips
├── requirements.txt        ←  Python dependencies
└── README.md               ←  This file
```

> Place `household_power_consumption.txt` in the same folder before running.

---

## 📦 Dataset

**UCI Individual Household Electric Power Consumption**

| Property | Value |
|---|---|
| Records | ~2,075,259 (minute-level readings) |
| Period | December 2006 – November 2010 |
| Frequency | 1 reading per minute |
| Missing values | ~1.25% (marked as `?`) |
| File format | Semicolon-delimited `.txt` |

### Columns

| Column | Unit | Description |
|---|---|---|
| `Date` | DD/MM/YYYY | Date of reading |
| `Time` | HH:MM:SS | Time of reading |
| `Global_active_power` | kW | Total active power consumed |
| `Global_reactive_power` | kVAR | Total reactive power |
| `Voltage` | V | RMS voltage |
| `Global_intensity` | A | Average current intensity |
| `Sub_metering_1` | Wh | Kitchen (dishwasher, oven, microwave) |
| `Sub_metering_2` | Wh | Laundry room (washer, dryer, fridge, light) |
| `Sub_metering_3` | Wh | Electric water heater & AC |

📥 **Download:** https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   DATA COLLECTION LAYER                     │
│       household_power_consumption.txt  (Smart Meter)        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   PROCESSING LAYER                          │
│   data_processor.py                                         │
│   • load_data()         — parse, clean, drop nulls         │
│   • engineer_features() — Energy_kWh, Power_factor, flags  │
│   • aggregate()         — resample to H / D / W            │
└─────────────┬───────────────────────────┬───────────────────┘
              │                           │
              ▼                           ▼
┌─────────────────────┐     ┌─────────────────────────────────┐
│   FORECASTING       │     │     ANOMALY DETECTION           │
│   ml_model.py       │     │     ml_model.py                 │
│                     │     │                                 │
│  Random Forest      │     │  Isolation Forest               │
│  • 200 estimators   │     │  • 150 estimators               │
│  • Lag features     │     │  • Contamination: 2%            │
│  • 24h ahead output │     │  • Anomaly score output         │
└──────────┬──────────┘     └──────────────┬──────────────────┘
           │                               │
           ▼                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     ALERT ENGINE                            │
│   alerts.py                                                 │
│   • evaluate_latest()   — current reading vs thresholds    │
│   • evaluate_bulk()     — scan full history for events     │
│   • optimisation_tips() — pattern-based recommendations    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   DASHBOARD LAYER                           │
│   app.py  — Streamlit (5 tabs, interactive sidebar)         │
│   • KPI Cards  • Plotly Charts  • ML Results  • Alert Log  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 ML Models

### 1. Demand Forecasting — Random Forest Regressor

Predicts hourly power consumption for the next 24 hours.

**Feature Engineering:**

| Feature Type | Features Created |
|---|---|
| Lag features | Power at t-1, t-2, t-3, t-6, t-12, t-24, t-48 hours |
| Rolling mean | 3h, 6h, 12h, 24h windows |
| Rolling std | 3h, 6h, 12h, 24h windows |
| Time features | Hour of day, day of week, month |
| Binary flags | `is_weekend`, `is_peak` (6 PM – 10 PM) |

**Training Setup:**
- **Split:** 80% train / 20% test — time-ordered (no data leakage)
- **Estimators:** 200 trees, max depth 12
- **Prediction:** Iterative 24-step ahead (each prediction feeds back as input)

**Typical Results (full dataset):**

| Metric | Value |
|---|---|
| MAE | ~0.25 kW |
| RMSE | ~0.38 kW |
| R² | ~0.65+ |

> Use 80–100% data sample in the sidebar for best accuracy.

---

### 2. Anomaly Detection — Isolation Forest

Flags abnormal readings that could indicate appliance faults, power surges, or meter errors.

**How it works:**
- Randomly partitions data using isolation trees
- Normal points require many splits to isolate → high score
- Anomalies are isolated quickly → low score
- Bottom 2% of scores are labelled as anomalies (adjustable 1–10%)

**Features used:** All 7 sensor columns — active power, reactive power, voltage, intensity, sub-metering 1/2/3

---

## 🔔 Alerts & Thresholds

All thresholds are **adjustable in real-time from the sidebar**.

| Condition | Default Threshold | Alert Level |
|---|---|---|
| Active Power | > 6.0 kW | 🔴 CRITICAL |
| Active Power | > 4.5 kW | 🟡 WARNING |
| Under-Voltage | < 210 V | 🔴 CRITICAL |
| Over-Voltage | > 250 V | 🟡 WARNING |
| High Current | > 25 A | 🟡 WARNING |
| Low Power Factor | < 0.85 | 🟡 WARNING |
| Normal Operation | All within range | 🟢 INFO |

### Auto-Generated Optimization Tips (Examples)

- ⚡ Shift heavy appliances to off-peak hours if evening usage is >40% higher than daytime
- 🍽️ Kitchen appliances are the top consumer — consider energy-efficient alternatives
- 🧺 Laundry/HVAC is dominant — use cold-water cycles and programmable thermostats
- 🌡️ Water heater/AC consuming excessively — consider a solar water heater
- 📉 Low power factor detected — capacitor correction recommended

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/energy-monitoring-system.git
cd energy-monitoring-system

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add the dataset
# Place household_power_consumption.txt in the project root folder
```

### Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.4.0
plotly>=5.18.0
```

---

## 🚀 Usage

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### Sidebar Controls

| Control | Purpose |
|---|---|
| 📂 File Upload | Upload dataset directly from browser |
| 📁 File Path | Point to dataset on your local machine |
| 🎚️ Data Sample % | 5–100% — lower is faster, higher is more accurate |
| 📅 Aggregation Frequency | Hourly (h), Daily (D), Weekly (W) |
| 📆 Date Range | Filter to a specific time period |
| ⚡ Alert Thresholds | Adjust warning and critical power levels live |

---

## 📊 Dashboard Tabs

### Tab 1 — 📊 Real-Time Dashboard
- Active power area chart with warning/critical threshold lines
- Sub-metering pie chart (Kitchen / Laundry / Water Heater / Other)
- Average power by hour of day bar chart
- Voltage distribution histogram

### Tab 2 — 🤖 AI Forecasting
- Model performance metrics (MAE, RMSE, R²)
- Actual vs Predicted overlay chart on test set
- Next 24-hour demand forecast table and chart
- Top 10 feature importances bar chart

### Tab 3 — 🚨 Anomaly Detection
- Scatter plot — normal (green dots) vs anomalies (red ✕)
- Anomaly score distribution histogram
- Table of the 5 most extreme anomalies with timestamps

### Tab 4 — 🔔 Alerts & Optimization
- Live alert status for the most recent reading
- Historical alert log with level filters (CRITICAL / WARNING)
- Alert frequency timeline chart
- AI-generated optimization tips

### Tab 5 — 📈 Deep Analytics
- **Heatmap:** Date × Hour power consumption matrix
- **Monthly trend:** Total energy per month bar chart
- **Weekday vs Weekend:** Overlaid hourly power profiles
- **Correlation matrix:** All sensor variable relationships

---

## 🌍 SDG Alignment

This project directly supports **UN Sustainable Development Goal 7: Affordable and Clean Energy**.

| SDG 7 Target | How This Project Contributes |
|---|---|
| ⚡ Energy Efficiency | Identifies waste patterns and suggests reduction strategies |
| 🌱 Sustainable Sources | Framework ready to integrate solar/wind generation data |
| 📉 Emission Reduction | Lower consumption directly reduces carbon footprint |
| 💰 Economic Growth | Cost savings through optimized peak/off-peak usage |
| 🏗️ Resilient Infrastructure | Anomaly detection prevents equipment damage and outages |

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | scikit-learn (Random Forest, Isolation Forest) |
| Visualization | Plotly Express & Graph Objects |
| Cloud Ready | AWS Lambda / Azure Functions compatible |
| Storage (architecture) | Amazon S3, DynamoDB |
| Notifications (architecture) | AWS SNS — SMS / Push |

---

## 🌐 Cloud Deployment

### Option 1 — Streamlit Community Cloud (Free & Easy)

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → select your repo → set main file to `app.py`
4. Upload the dataset via the sidebar file uploader once deployed

### Option 2 — AWS (Production Architecture)

```
S3 Bucket       → Store raw dataset files
Lambda Function → Run data_processor.py + ml_model.py as microservices
API Gateway     → REST API endpoints for dashboard queries
DynamoDB        → Persist alert logs and forecast results
SNS             → Push/SMS notifications for CRITICAL alerts
EC2 / ECS       → Host Streamlit application container
```

> The `data_processor.py` and `ml_model.py` modules are stateless and importable — they can be wrapped in a Lambda handler with minimal modification.

---

## 📄 License

This project is licensed under the **MIT License** — free for academic, personal, and commercial use.

```
MIT License — Copyright (c) 2024 Nikesh Sinha
Permission is hereby granted, free of charge, to any person obtaining
a copy of this software to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software.
```

---
