"""
FastAPI Backend Server for Energy Consumption Pattern Mining
Exposes clean RESTful API endpoints for the analytical and machine learning pipeline,
and serves the modern HTML/CSS/JS frontend.
"""
import io
import os
import sys
import numpy as np
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import energy_mining as em

HERE = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(HERE, "frontend")
SAMPLE_FILE = os.path.join(HERE, "sample_upload.csv")

app = FastAPI(
    title="Energy Consumption Pattern Mining API",
    description="Full-stack AI & Data Mining platform for electrical load intelligence",
    version="2.0.0"
)

# Enable CORS for local development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory pipeline cache
state = {
    "source": None,
    "raw_df": None,
    "results": None,
    "trained_model": None,
    "model_features": None
}


def sanitize(val):
    """Recursively converts numpy, pandas, and datetime objects into JSON-safe types."""
    if val is None:
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    if isinstance(val, (pd.Timestamp, np.datetime64)):
        return str(val)
    if isinstance(val, np.ndarray):
        return [sanitize(x) for x in val]
    if isinstance(val, pd.Series):
        return {str(k): sanitize(v) for k, v in val.items()}
    if isinstance(val, pd.DataFrame):
        clean_df = val.replace({np.nan: None})
        return clean_df.to_dict(orient="records")
    if isinstance(val, dict):
        return {str(k): sanitize(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [sanitize(x) for x in val]
    return val


def train_live_classifier(df):
    """Train a fast classifier and preserve model + features for live prediction."""
    from sklearn.ensemble import RandomForestClassifier
    d, cuts = em.add_level(df)
    d = d.copy()
    d["lag_24"] = d["kw"].shift(24)
    d["lag_168"] = d["kw"].shift(168)
    d = d.dropna()
    feats = ["hour", "dow", "month", "weekend", "temp_c", "peak", "lag_24", "lag_168"]
    split = int(len(d) * 0.8)
    tr = d.iloc[:split]
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=em.SEED, n_jobs=-1)
    rf.fit(tr[feats], tr["level"])
    return rf, feats, cuts


def initialize_pipeline(raw_df=None, source_label=None):
    """Run data mining pipeline and populate the state cache."""
    print("Executing Energy Mining Pipeline...")
    if raw_df is None:
        raw_df, source_label = em.load_data()
    results = em.run_all(raw_df, source_label)
    try:
        model, feats, cuts = train_live_classifier(results["df"])
    except Exception as e:
        print(f"Classifier live cache warning: {e}")
        model, feats, cuts = None, None, results["clf"]["cuts"]
    
    state["raw_df"] = raw_df
    state["source"] = source_label
    state["results"] = results
    state["trained_model"] = model
    state["model_features"] = feats
    print(f"Pipeline ready. Source: {source_label}, Records: {len(results['df'])}")


# Startup event
@app.on_event("startup")
def startup_event():
    os.makedirs(FRONTEND_DIR, exist_ok=True)
    initialize_pipeline()


# =====================================================================
# API Endpoints
# =====================================================================

@app.get("/api/status")
def get_status():
    if state["results"] is None:
        return {"ready": False}
    df = state["results"]["df"]
    return {
        "ready": True,
        "source": state["source"],
        "records": len(df),
        "date_min": df.index.min().strftime("%Y-%m-%d"),
        "date_max": df.index.max().strftime("%Y-%m-%d"),
        "has_temp": "temp_c" in df.columns,
        "is_proxy": "proxy" in str(state["source"]).lower()
    }


@app.get("/api/overview")
def get_overview():
    if state["results"] is None:
        raise HTTPException(status_code=503, detail="Pipeline is warming up.")
    
    R = state["results"]
    df = R["df"]
    patterns = R["patterns"]
    clusters = R["clusters"]
    anomalies = R["anomalies"]
    
    tot_energy = float(df["kw"].sum())
    avg_load = float(df["kw"].mean())
    peak_load = float(df["kw"].max())
    peak_share = float(df.loc[df["peak"] == 1, "kw"].sum() / tot_energy * 100) if tot_energy > 0 else 0
    anomaly_count = int(anomalies["anomaly"].sum())
    spikes = int((anomalies.loc[anomalies["anomaly"], "type"] == "Spike").sum())
    drops = int((anomalies.loc[anomalies["anomaly"], "type"] == "Drop").sum())
    
    standby_baseline = float(df.groupby("date")["kw"].min().median())
    standby_daily_kwh = float(standby_baseline * 24)
    
    seas = patterns["seasonal"]
    high_season = str(seas.idxmax())
    high_season_kw = float(seas.max())
    low_season = str(seas.idxmin())
    low_season_kw = float(seas.min())
    
    top_hours = patterns["hourly"].nlargest(3)
    top_hours_list = [{"hour": int(h), "kw": round(float(v), 2)} for h, v in top_hours.items()]
    
    # Financial & carbon estimates (standard commercial/residential mix: $0.16/kWh, 0.385 kg CO2/kWh)
    cost_usd = round(tot_energy * 0.16, 2)
    carbon_kg = round(tot_energy * 0.385, 1)

    return sanitize({
        "source": state["source"],
        "records": len(df),
        "days": len(df["date"].unique()),
        "date_range": {
            "min": df.index.min().strftime("%Y-%m-%d"),
            "max": df.index.max().strftime("%Y-%m-%d")
        },
        "kpis": {
            "avg_load_kw": round(avg_load, 2),
            "peak_load_kw": round(peak_load, 2),
            "total_energy_kwh": round(tot_energy, 1),
            "peak_window_share_pct": round(peak_share, 1),
            "anomaly_count": anomaly_count,
            "anomaly_spikes": spikes,
            "anomaly_drops": drops,
            "standby_baseline_kw": round(standby_baseline, 2),
            "standby_daily_kwh": round(standby_daily_kwh, 1),
            "high_season": high_season,
            "high_season_kw": round(high_season_kw, 2),
            "low_season": low_season,
            "low_season_kw": round(low_season_kw, 2),
            "top_hours": top_hours_list,
            "est_cost_usd": cost_usd,
            "carbon_kg": carbon_kg
        },
        "seasons": ["Winter", "Spring", "Summer", "Autumn"]
    })


@app.get("/api/patterns")
def get_patterns(seasons: Optional[List[str]] = Query(None)):
    if state["results"] is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    df = state["results"]["df"]
    if seasons and len(seasons) > 0:
        df = df[df["season"].isin(seasons)]
        if df.empty:
            df = state["results"]["df"]

    # 1. Day of Week x Hour Heatmap Matrix
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    hp = df.pivot_table(index="dayname", columns="hour", values="kw", aggfunc="mean").reindex(dow_order)
    
    matrix = []
    for day in dow_order:
        row_vals = []
        for h in range(24):
            val = hp.loc[day, h] if day in hp.index and h in hp.columns else 0.0
            row_vals.append(round(float(val), 3) if pd.notna(val) else 0.0)
        matrix.append({"day": day, "values": row_vals})

    # 2. Weekday vs Weekend Hourly Profile
    wk = df.groupby(["weekend", "hour"])["kw"].mean().unstack(0)
    weekday_vals = [round(float(wk[0].get(h, 0)), 3) for h in range(24)] if 0 in wk.columns else [0.0]*24
    weekend_vals = [round(float(wk[1].get(h, 0)), 3) for h in range(24)] if 1 in wk.columns else [0.0]*24

    # 3. Monthly Profile
    month_names = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
                   7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}
    m = df.groupby("month")["kw"].mean()
    monthly_data = [{"month": month_names.get(int(k), str(k)), "kw": round(float(v), 3)} for k, v in m.items()]

    # 4. Seasonal Profile
    seas = df.groupby("season")["kw"].mean()
    seasonal_data = [{"season": str(k), "kw": round(float(v), 3)} for k, v in seas.items()]

    # 5. Temperature vs Load Scatter (downsampled for fast browser rendering)
    scatter_sample = df.sample(min(800, len(df)), random_state=42)
    scatter_data = [
        {
            "temp_c": round(float(r["temp_c"]), 1),
            "kw": round(float(r["kw"]), 2),
            "daypart": str(r["daypart"]),
            "hour": int(r["hour"]),
            "season": str(r["season"])
        }
        for _, r in scatter_sample.iterrows()
    ]

    return sanitize({
        "heatmap": {
            "days": dow_order,
            "hours": list(range(24)),
            "rows": matrix,
            "min_kw": round(float(df["kw"].min()), 2),
            "max_kw": round(float(df["kw"].max()), 2),
            "mean_kw": round(float(df["kw"].mean()), 2)
        },
        "weekday_vs_weekend": {
            "hours": list(range(24)),
            "weekday": weekday_vals,
            "weekend": weekend_vals
        },
        "monthly": monthly_data,
        "seasonal": seasonal_data,
        "temp_scatter": scatter_data
    })


@app.get("/api/clusters")
def get_clusters():
    if state["results"] is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    C = state["results"]["clusters"]
    
    # K-Means Evaluation (Elbow Inertia & Silhouette Scores)
    scores = C["scores"].to_dict(orient="records")
    for s in scores:
        s["k"] = int(s["k"])
        s["inertia"] = round(float(s["inertia"]), 1)
        s["silhouette"] = round(float(s["silhouette"]), 3)

    # Centroids per cluster across 24 hours
    cen_df = C["centers"]
    centroids = {}
    for cluster_name, row in cen_df.iterrows():
        centroids[str(cluster_name)] = [round(float(v), 3) for v in row.values]

    # Cluster distribution
    dist_s = C["labels"].value_counts()
    tot_days = len(C["labels"])
    distribution = [
        {
            "cluster": str(c_name),
            "days": int(cnt),
            "percentage": round(float(cnt) / tot_days * 100, 1),
            "avg_kw": round(float(cen_df.loc[c_name].mean()), 2)
        }
        for c_name, cnt in dist_s.items()
    ]

    # PCA 2D coordinates (downsampled if huge)
    pca_df = C["pca"].assign(cluster=C["labels"])
    pca_sample = pca_df.sample(min(500, len(pca_df)), random_state=42)
    pca_points = [
        {
            "date": str(idx)[:10],
            "pc1": round(float(row["PC1"]), 2),
            "pc2": round(float(row["PC2"]), 2),
            "cluster": str(row["cluster"])
        }
        for idx, row in pca_sample.iterrows()
    ]

    return sanitize({
        "chosen_k": int(C["k"]),
        "evaluation": scores,
        "centroids": {
            "hours": list(range(24)),
            "curves": centroids
        },
        "distribution": distribution,
        "pca_points": pca_points
    })


@app.get("/api/classification")
def get_classification():
    if state["results"] is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    K = state["results"]["clf"]
    q1, q2 = K["cuts"]
    
    # Model comparison results
    model_comparison = K["results"].to_dict(orient="records")
    for m in model_comparison:
        m["accuracy"] = round(float(m["accuracy"]) * 100, 1)
        m["macro_f1"] = round(float(m["macro_f1"]) * 100, 1)

    # Confusion matrix
    conf_df = K["confusion"]
    labels = ["Low", "Medium", "High"]
    confusion_matrix_data = {
        "labels": labels,
        "matrix": [[int(conf_df.loc[actual, pred]) for pred in labels] for actual in labels]
    }

    # Feature Importance ranking
    feat_imp = [
        {"feature": str(f), "importance": round(float(v) * 100, 2)}
        for f, v in K["importance"].items()
    ]

    # Classification report breakdown
    report = K["report"]
    report_summary = {}
    for lab in ["Low", "Medium", "High"]:
        if lab in report:
            report_summary[lab] = {
                "precision": round(float(report[lab]["precision"]) * 100, 1),
                "recall": round(float(report[lab]["recall"]) * 100, 1),
                "f1_score": round(float(report[lab]["f1-score"]) * 100, 1),
                "support": int(report[lab]["support"])
            }

    return sanitize({
        "tertiles": {
            "q1_low": round(float(q1), 2),
            "q2_high": round(float(q2), 2),
            "description": f"Low (<= {round(float(q1), 2)} kW), Medium ({round(float(q1), 2)} - {round(float(q2), 2)} kW), High (> {round(float(q2), 2)} kW)"
        },
        "models": model_comparison,
        "confusion_matrix": confusion_matrix_data,
        "feature_importance": feat_imp,
        "report": report_summary
    })


class PredictionInput(BaseModel):
    hour: int
    dow: int
    month: int
    weekend: int
    temp_c: float
    peak: int
    lag_24: float
    lag_168: float


@app.post("/api/predict")
def predict_load(req: PredictionInput):
    """Run real-time ML inference on user input parameters."""
    model = state["trained_model"]
    feats = state["model_features"]
    cuts = state["results"]["clf"]["cuts"] if state["results"] else (0.5, 1.2)
    
    if model is None:
        # Fallback heuristic based on cuts
        val = req.lag_24 * 0.5 + req.lag_168 * 0.3 + (0.8 if req.peak else 0.2)
        if val <= cuts[0]:
            pred = "Low"
            probs = {"Low": 0.75, "Medium": 0.20, "High": 0.05}
        elif val <= cuts[1]:
            pred = "Medium"
            probs = {"Low": 0.15, "Medium": 0.70, "High": 0.15}
        else:
            pred = "High"
            probs = {"Low": 0.05, "Medium": 0.20, "High": 0.75}
        return {"level": pred, "probabilities": probs, "tertiles": cuts}

    # Format vector
    input_row = pd.DataFrame([{
        "hour": req.hour,
        "dow": req.dow,
        "month": req.month,
        "weekend": req.weekend,
        "temp_c": req.temp_c,
        "peak": req.peak,
        "lag_24": req.lag_24,
        "lag_168": req.lag_168
    }])[feats]
    
    pred_level = str(model.predict(input_row)[0])
    prob_vals = model.predict_proba(input_row)[0]
    classes = [str(c) for c in model.classes_]
    prob_dict = {cls: round(float(prob_vals[i]) * 100, 1) for i, cls in enumerate(classes)}

    factors = []
    if req.peak == 1:
        factors.append("Operating within evening peak pricing window (17:00-21:59).")
    if req.temp_c < 10:
        factors.append("Cold ambient temperature inducing supplemental heating demand.")
    elif req.temp_c > 26:
        factors.append("High ambient temperature triggering cooling/HVAC compressors.")
    if req.lag_24 > cuts[1]:
        factors.append("High historical consumption persisted at this exact hour yesterday.")
    if not factors:
        factors.append("Nominal baseline conditions observed across calendar and weather variables.")

    return sanitize({
        "level": pred_level,
        "probabilities": prob_dict,
        "tertiles": {"q1": round(float(cuts[0]), 2), "q2": round(float(cuts[1]), 2)},
        "factors": factors
    })


@app.get("/api/association-rules")
def get_association_rules():
    if state["results"] is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    rules_df = state["results"]["rules"]
    if rules_df.empty:
        return {"rules": [], "empty_reason": "No association rules met support (0.03) and confidence (0.6) thresholds."}

    records = []
    for idx, r in rules_df.iterrows():
        records.append({
            "id": idx + 1,
            "rule": str(r["rule"]),
            "support": round(float(r["support"]) * 100, 1),
            "confidence": round(float(r["confidence"]) * 100, 1),
            "lift": round(float(r["lift"]), 2)
        })

    return sanitize({"rules": records})


@app.get("/api/anomalies")
def get_anomalies(limit: int = 50):
    if state["results"] is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    A = state["results"]["anomalies"]
    anom_rows = A[A["anomaly"]].copy()
    
    total_anomalies = int(len(anom_rows))
    spikes = int((anom_rows["type"] == "Spike").sum())
    drops = int((anom_rows["type"] == "Drop").sum())

    # Top anomalies sorted by absolute robust z-score
    top_50 = (
        anom_rows[["kw", "robust_z", "type", "hour", "dayname", "temp_c"]]
        .sort_values("robust_z", key=abs, ascending=False)
        .head(limit)
    )
    
    table_data = []
    for dt, row in top_50.iterrows():
        table_data.append({
            "timestamp": dt.strftime("%Y-%m-%d %H:%M"),
            "kw": round(float(row["kw"]), 2),
            "robust_z": round(float(row["robust_z"]), 2),
            "type": str(row["type"]),
            "hour": f"{int(row['hour']):02d}:00",
            "dayname": str(row["dayname"]),
            "temp_c": round(float(row["temp_c"]), 1) if pd.notna(row["temp_c"]) else None
        })

    # Time series snippet (last 300 data points for smooth line rendering with anomaly markers)
    recent = A.iloc[-300:]
    ts_data = []
    for dt, row in recent.iterrows():
        ts_data.append({
            "t": dt.strftime("%m-%d %H:%M"),
            "kw": round(float(row["kw"]), 2),
            "anomaly": bool(row["anomaly"]),
            "type": str(row["type"]) if row["anomaly"] else None
        })

    return sanitize({
        "summary": {
            "total": total_anomalies,
            "spikes": spikes,
            "drops": drops,
            "rate_pct": round(total_anomalies / len(A) * 100, 2)
        },
        "top_table": table_data,
        "recent_series": ts_data
    })


@app.get("/api/insights")
def get_insights():
    if state["results"] is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    raw_insights = state["results"]["insights"]
    categorized = []
    
    categories = [
        {"title": "Peak Demand Mitigation", "icon": "zap", "tag": "High Impact", "color": "amber"},
        {"title": "Peak Hourly Usage Target", "icon": "clock", "tag": "Scheduling", "color": "cyan"},
        {"title": "Standby Vampire Load", "icon": "power", "tag": "Continuous Saving", "color": "emerald"},
        {"title": "Seasonal Weather Vulnerability", "icon": "thermometer", "tag": "HVAC Efficiency", "color": "blue"},
        {"title": "Weekend Consumption Discrepancy", "icon": "calendar", "tag": "Behavioral", "color": "purple"},
        {"title": "Load Profile Cluster Targeting", "icon": "grid", "tag": "Demand Response", "color": "indigo"},
        {"title": "Fault & Anomaly Remediation", "icon": "alert-triangle", "tag": "Maintenance", "color": "rose"}
    ]
    
    for i, ins in enumerate(raw_insights):
        meta = categories[i % len(categories)]
        categorized.append({
            "id": i + 1,
            "category": meta["title"],
            "icon": meta["icon"],
            "tag": meta["tag"],
            "color": meta["color"],
            "text": ins
        })

    return sanitize({"insights": categorized})


class SimulationRequest(BaseModel):
    peak_shift_pct: float = 20.0       # 0 - 50% shifted from peak (17-21) to off-peak
    standby_reduction_pct: float = 25.0 # 0 - 50% standby vampire load reduced
    peak_rate: float = 0.28            # Peak tariff $/kWh
    offpeak_rate: float = 0.12         # Off-peak tariff $/kWh
    base_rate: float = 0.16            # Average baseline tariff $/kWh


@app.post("/api/simulate")
def simulate_savings(req: SimulationRequest):
    """Interactive What-If energy and cost saving scenario simulation."""
    if state["results"] is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    df = state["results"]["df"]
    tot_kwh = float(df["kw"].sum())
    days = max(1, len(df["date"].unique()))
    annual_multiplier = 365.0 / days
    
    annual_base_kwh = tot_kwh * annual_multiplier
    baseline_cost = annual_base_kwh * req.base_rate
    
    # 1. Peak shifting simulation
    peak_kwh = float(df.loc[df["peak"] == 1, "kw"].sum()) * annual_multiplier
    shifted_kwh = peak_kwh * (req.peak_shift_pct / 100.0)
    # Peak tariff differential savings: shifted kWh moves from peak_rate to offpeak_rate
    peak_dollar_savings = shifted_kwh * (req.peak_rate - req.offpeak_rate)
    
    # 2. Standby reduction simulation
    standby_kw = float(df.groupby("date")["kw"].min().median())
    standby_annual_kwh = (standby_kw * 24 * 365.0)
    standby_kwh_saved = standby_annual_kwh * (req.standby_reduction_pct / 100.0)
    standby_dollar_savings = standby_kwh_saved * req.offpeak_rate
    
    # Total impact
    total_dollar_saved = round(peak_dollar_savings + standby_dollar_savings, 2)
    total_kwh_saved = round(standby_kwh_saved, 1) # shifted kWh is not destroyed, but standby is
    new_annual_cost = round(baseline_cost - total_dollar_saved, 2)
    co2_saved_kg = round(standby_kwh_saved * 0.385, 1)
    trees_equivalent = round(co2_saved_kg / 21.77, 1)  # 1 mature tree absorbs ~21.77 kg CO2/year

    return sanitize({
        "baseline": {
            "annual_kwh": round(annual_base_kwh, 1),
            "annual_cost_usd": round(baseline_cost, 2)
        },
        "optimized": {
            "annual_kwh": round(annual_base_kwh - standby_kwh_saved, 1),
            "annual_cost_usd": new_annual_cost
        },
        "savings": {
            "annual_dollar_saved": total_dollar_saved,
            "pct_cost_saved": round((total_dollar_saved / baseline_cost) * 100, 1) if baseline_cost > 0 else 0,
            "annual_kwh_eliminated": total_kwh_saved,
            "annual_kwh_shifted": round(shifted_kwh, 1),
            "co2_avoided_kg": co2_saved_kg,
            "trees_planted_equiv": trees_equivalent
        }
    })


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Upload custom CSV, validate, and retrain pipeline."""
    try:
        contents = await file.read()
        raw_df, source = em.load_csv(io.BytesIO(contents))
        initialize_pipeline(raw_df, f"Uploaded: {file.filename}")
        return {
            "success": True,
            "message": f"Successfully loaded {len(raw_df)} records from {file.filename}",
            "records": len(raw_df),
            "source": state["source"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/load-sample")
def load_sample_file():
    """Quick load built-in sample_upload.csv dataset."""
    if not os.path.exists(SAMPLE_FILE):
        raise HTTPException(status_code=404, detail="sample_upload.csv not found")
    try:
        raw_df, source = em.load_csv(SAMPLE_FILE)
        initialize_pipeline(raw_df, "Sample Upload CSV (60 Days)")
        return {
            "success": True,
            "message": "Loaded sample_upload.csv successfully",
            "records": len(raw_df),
            "source": state["source"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/reset")
def reset_to_default():
    """Reset pipeline to default dataset (UCI or Synthetic)."""
    initialize_pipeline()
    return {
        "success": True,
        "message": "Reset to default benchmark dataset",
        "source": state["source"]
    }


# =====================================================================
# Serve Frontend Single Page Application
# =====================================================================

# Serve root index.html
@app.get("/")
def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend files will be created in frontend/ directory."}

# Mount static assets
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
