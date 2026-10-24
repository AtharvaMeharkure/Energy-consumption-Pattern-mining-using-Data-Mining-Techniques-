# ⚡ VoltPulse AI — Energy Consumption Pattern Mining Platform

An enterprise-grade, full-stack energy intelligence and pattern mining platform with an interactive modern **HTML5 / CSS3 / Vanilla JS** frontend and a high-performance **FastAPI + Scikit-Learn** backend.

---

## 🚀 Quick Start (Full-Stack Platform)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the Complete Web App**:
   ```bash
   python run_app.py
   ```
   *Automatically launches the backend server and opens **`http://127.0.0.1:8000`** in your browser.*

   Alternatively, launch via Uvicorn:
   ```bash
   python server.py
   ```

3. **Legacy Modes (Optional)**:
   - Streamlit dashboard: `streamlit run app.py`
   - Console pipeline report: `python energy_mining.py`

---

## 🌟 Key Features & Analytical Architecture

### 1. Executive Dashboard & KPIs
- **Diurnal Load Tracking**: Average load, peak load, and cumulative energy consumption.
- **Standby Vampire Load Analysis**: Computes the daily minimum standby baseline (`kW`) and always-on waste (`kWh/day`).
- **Peak Window Demand Ratio**: Isolates the critical 17:00–21:59 peak window and measures peak load concentration.
- **Financial & Carbon Modeling**: Real-time utility billing estimates and carbon footprint calculations.

### 2. Temporal Pattern Discovery
- **Day of Week × Hour Matrix**: Interactive 2D heatmap isolating high-load demand clusters.
- **Weekday vs. Weekend Load Curves**: Diurnal hourly profiles highlighting lifestyle shifts.
- **Monthly & Seasonal Trends**: Measures temperature-driven heating and cooling swings.
- **Ambient Weather Correlation**: Scatter plot mapping active power draw against ambient temperature (°C).

### 3. Unsupervised Load Profile Clustering
- **K-Means Daily Profile Mining**: Clusters 24-hour load shapes into archetypes (e.g. low base, evening peak, midday weekend).
- **Elbow Curve & Silhouette Score**: Automatic selection of optimal cluster count $k \ge 3$.
- **PCA 2D Latent Space Projection**: High-dimensional variance projection on principal components PC1 and PC2.

### 4. Machine Learning Classification & Live Inference
- **Multi-Class Consumption Tiers**: Tertile thresholds (*Low*, *Medium*, *High*).
- **Leak-Free Lag Features**: Uses autoregressive 24-hour and 168-hour (weekly) past lags.
- **Model Benchmark**: Random Forest Classifier vs. Decision Tree with Accuracy, Macro F1, Precision, and Recall.
- **Interactive Live Predictor**: Test custom scenarios (hour, temperature, day, historical lags) to receive immediate ML inference with class probabilities.

### 5. Association Rule Mining (Apriori Algorithm)
- Discovers multi-condition dependencies linking environmental conditions, dayparts, and seasonal triggers to consumption tiers.
- Filterable and sortable by **Lift**, **Confidence**, and **Support**.

### 6. Anomaly & Fault Detection
- **Dual Ensemble Detection**: Combines scikit-learn's **Isolation Forest** with **Robust Median Absolute Deviation (MAD) Z-Scores**.
- Distinguishes between **Spikes** (left-on appliances, compressor faults) and **Drops** (outages, sensor dropouts).
- Audit log of the top 50 anomalous events with automated remediation recommendations.

### 7. AI Energy-Saving Insights & What-If Scenario Simulator
- Synthesizes actionable recommendations across HVAC, standby loads, and peak tariff mitigation.
- **Interactive ROI Calculator**:
  - Peak load shifting slider (17:00–21:59).
  - Standby vampire load reduction slider.
  - Computes annual dollar savings, % bill cut, eliminated kWh, avoided $\text{CO}_2$, and equivalent trees planted.

---

## 📂 Project Structure

```text
energy/
├── frontend/                     # Modern Web UI (HTML5, CSS3, JavaScript)
│   ├── index.html                # Single-page application structure
│   ├── styles.css                # Dark glassmorphic design system & animations
│   └── app.js                    # Chart.js/Plotly.js rendering, ML inference & simulator
├── server.py                     # FastAPI backend & RESTful analytics API
├── run_app.py                    # One-click launcher script
├── energy_mining.py              # Core data mining & ML algorithms
├── app.py                        # Streamlit dashboard
├── sample_upload.csv             # 60-day sample dataset with datetime, kw, temp_c
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
```

---

## 📊 Dataset Ingestion & Uploads

- **Built-in Benchmark**: Default realistic 2-year hourly synthetic dataset with simulated diurnal cycles, weather, heating/cooling, and anomalies.
- **Bundled Real Sample**: Click the **Sample CSV** button in the header to load `sample_upload.csv`.
- **Custom CSV Upload**: Click **Upload CSV** or drag-and-drop your dataset. Requirements:
  - Any sampling interval (automatically resampled to hourly mean).
  - A timestamp column (`datetime`, `Timestamp`, or `Date` + `Time`).
  - An active power column (`kw`, `EnergyConsumption`, `Global_active_power`).
  - Optional weather column (`temp_c` or `Temperature`).
