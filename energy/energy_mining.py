"""
Energy Consumption Pattern Mining - core pipeline
-------------------------------------------------
Covers: preprocessing, temporal patterns, clustering, classification,
association rules, anomaly detection and energy-saving insights.

Data: put UCI "household_power_consumption.txt" next to this file to use real
data. Otherwise a realistic synthetic hourly dataset is generated.
Optional: weather.csv with columns [datetime, temp_c] (e.g. from Open-Meteo).
"""
import io
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from mlxtend.frequent_patterns import apriori, association_rules

HERE = os.path.dirname(os.path.abspath(__file__))
UCI_FILE = os.path.join(HERE, "household_power_consumption.txt")
WEATHER_FILE = os.path.join(HERE, "weather.csv")
SEED = 42


# ----------------------------------------------------------------- 1. DATA
def _synthetic(days=730, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=days * 24, freq="h")
    doy = idx.dayofyear.values
    hour = idx.hour.values
    weekend = (idx.dayofweek >= 5).astype(int)
    temp = 15 - 10 * np.cos(2 * np.pi * (doy - 15) / 365) + 4 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 2, len(idx))
    morning = np.exp(-((hour - 7.5) ** 2) / 3)
    evening = np.exp(-((hour - 19.5) ** 2) / 4)
    midday = np.exp(-((hour - 13) ** 2) / 8) * weekend
    heating = np.clip(12 - temp, 0, None) * 0.07 * (0.4 + evening + morning)
    cooling = np.clip(temp - 26, 0, None) * 0.10
    kw = 0.35 + 0.9 * morning * (1 - 0.4 * weekend) + 1.6 * evening + 0.8 * midday + heating + cooling
    kw = kw * rng.lognormal(0, 0.12, len(idx))
    kw = pd.Series(kw, index=idx)
    # inject anomalies: night spikes and outage-like drops
    spikes = rng.choice(len(idx), 25, replace=False)
    kw.iloc[spikes] *= rng.uniform(3, 5, 25)
    drops = rng.choice(len(idx), 12, replace=False)
    kw.iloc[drops] = 0.02
    return pd.DataFrame({"kw": kw, "temp_c": temp}), "synthetic"


def load_data():
    """Return hourly dataframe indexed by datetime with columns kw, temp_c, plus source label."""
    if os.path.exists(UCI_FILE):
        raw = pd.read_csv(UCI_FILE, sep=";", na_values="?", low_memory=False,
                          usecols=["Date", "Time", "Global_active_power"])
        raw["dt"] = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H:%M:%S")
        s = raw.set_index("dt")["Global_active_power"].astype(float)
        df = s.resample("h").mean().to_frame("kw")      # average kW per hour (~kWh)
        source = "UCI household"
    else:
        df, source = _synthetic()
    return _add_temp(df, source)


def _add_temp(df, source, use_file=True):
    if "temp_c" not in df.columns:
        if use_file and os.path.exists(WEATHER_FILE):
            w = pd.read_csv(WEATHER_FILE, parse_dates=["datetime"]).set_index("datetime")["temp_c"]
            df = df.join(w, how="left")
        else:  # seasonal proxy - clearly NOT real weather
            doy = df.index.dayofyear.values
            df["temp_c"] = 12 - 9 * np.cos(2 * np.pi * (doy - 15) / 365)
            source += " + seasonal temp proxy"
    return df, source


def load_csv(file, min_days=30):
    """Load a user-uploaded CSV. Needs a datetime column (or Date + Time) and a power column in kW.
    Optional temperature column. Any sampling interval; resampled to hourly mean."""
    data = file.read() if hasattr(file, "read") else open(file, "rb").read()
    if isinstance(data, bytes):  # utf-8-sig strips the Excel BOM that would break the first column name
        data = data.decode("utf-8-sig", errors="replace")
    raw = pd.read_csv(io.StringIO(data), sep=None, engine="python", na_values=["?", "", "NA", "nan", "NaN"])
    raw.columns = [str(c).strip() for c in raw.columns]
    cols = {str(c).strip().lower(): c for c in raw.columns}
    dcol = next((cols[k] for k in ("datetime", "timestamp", "date_time", "time_stamp") if k in cols), None)
    if dcol is not None:
        txt = raw[dcol].astype(str)
    elif "date" in cols and "time" in cols:
        txt = raw[cols["date"]].astype(str) + " " + raw[cols["time"]].astype(str)
    else:
        dcol = raw.columns[0]
        txt = raw[dcol].astype(str)
    dt = pd.to_datetime(txt, errors="coerce")
    if dt.isna().mean() > 0.5:
        dt = pd.to_datetime(txt, errors="coerce", dayfirst=True)
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    if dt.isna().mean() > 0.5:
        raise ValueError("Could not read dates. Use a 'datetime' column (e.g. 2024-01-31 18:00) or Date + Time columns.")
    norm = {c: str(c).lower().replace("_", "").replace(" ", "") for c in raw.columns}

    def find(exact, contains=(), avoid=()):
        for c, n in norm.items():
            if n in exact:
                return c
        for sub in contains:
            for c, n in norm.items():
                if sub in n and not any(x in n for x in avoid):
                    return c
        return None

    tcol = find(("tempc", "temperature", "temperaturec", "temp"), ("temp",), ("set",))
    pcol = find(("kw", "power", "globalactivepower", "consumption", "energyconsumption", "usage", "load", "value"),
                ("consumption", "kw", "power", "energy"), ("renewable", "solar"))
    if pcol is not None and pd.to_numeric(raw[pcol], errors="coerce").notna().mean() < 0.5:
        pcol = None
    if pcol is None:
        skip = {dcol, tcol, cols.get("date"), cols.get("time")}
        num = [c for c in raw.columns if c not in skip and pd.to_numeric(raw[c], errors="coerce").notna().mean() > 0.5]
        if not num:
            raise ValueError("No numeric power column found. Add a column named 'kw' (power in kW).")
        pcol = num[0]
    out = pd.DataFrame({"kw": pd.to_numeric(raw[pcol], errors="coerce").values}, index=dt)
    if tcol is not None:
        out["temp_c"] = pd.to_numeric(raw[tcol], errors="coerce").values
    out = out[out.index.notna()].sort_index()
    out = out[~out.index.duplicated()].resample("h").mean()
    out = out.dropna(subset=["kw"])
    if len(out) < 24 * min_days:
        raise ValueError(f"Need at least {min_days} days of data ({24 * min_days} hourly readings); found {len(out)}.")
    return _add_temp(out, "uploaded CSV", use_file=False)


def preprocess(df):
    """Clean: interpolate short gaps, drop long ones, remove duplicates, add features."""
    df = df[~df.index.duplicated()].asfreq("h")
    df["kw"] = df["kw"].interpolate(limit=6)
    df["temp_c"] = df["temp_c"].interpolate(limit=6)
    df = df.dropna().copy()
    df = df[df["kw"] >= 0]
    df["date"] = df.index.normalize()
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["dayname"] = df.index.day_name()
    df["month"] = df.index.month
    df["weekend"] = (df["dow"] >= 5).astype(int)
    df["season"] = df["month"].map({12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
                                   6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"})
    df["daypart"] = pd.cut(df["hour"], [-1, 5, 11, 17, 21, 23],
                           labels=["Night", "Morning", "Afternoon", "Evening", "Late"]).astype(str)
    df["daypart"] = df["daypart"].replace("Late", "Night")
    df["peak"] = df["hour"].between(17, 21).astype(int)
    return df


# ------------------------------------------------------------ 2. PATTERNS
def temporal_patterns(df):
    return {
        "hour_dow": df.pivot_table(index="dayname", columns="hour", values="kw", aggfunc="mean")
                      .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
        "hourly": df.groupby("hour")["kw"].mean(),
        "weekday_vs_weekend": df.groupby(["weekend", "hour"])["kw"].mean().unstack(0).rename(columns={0: "Weekday", 1: "Weekend"}),
        "monthly": df.groupby("month")["kw"].mean(),
        "seasonal": df.groupby("season")["kw"].mean(),
    }


# ------------------------------------------------------------- 3. CLUSTER
def daily_profiles(df):
    p = df.pivot_table(index="date", columns="hour", values="kw")
    return p.dropna(thresh=20).apply(lambda r: r.fillna(r.mean()), axis=1)


def choose_k(X, ks=range(2, 8)):
    out = []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=SEED).fit(X)
        out.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(out)


def cluster_days(profiles, k=None):
    X = StandardScaler().fit_transform(profiles.values)
    scores = choose_k(X)
    k = k or int(scores[scores["k"] >= 3].sort_values("silhouette", ascending=False).iloc[0]["k"])  # k>=3 so clusters are informative
    km = KMeans(k, n_init=10, random_state=SEED).fit(X)
    labels = km.labels_
    # name clusters by mean level, low -> high
    order = profiles.groupby(labels).mean().mean(axis=1).sort_values().index.tolist()
    names = {c: f"Cluster {i + 1} ({['lowest', 'low', 'mid', 'high', 'highest', 'peak'][min(i, 5)] if k > 2 else ['low', 'high'][i]} use)"
             for i, c in enumerate(order)}
    lab = pd.Series(labels, index=profiles.index).map(names)
    pca = PCA(2, random_state=SEED).fit_transform(X)
    centers = profiles.groupby(lab.values).mean()
    return {"labels": lab, "scores": scores, "k": k, "pca": pd.DataFrame(pca, index=profiles.index, columns=["PC1", "PC2"]),
            "centers": centers}


def household_features(df):
    """Per-household feature table for clustering households (needs a 'household' column)."""
    g = df.groupby("household")["kw"]
    return pd.DataFrame({"mean": g.mean(), "peak_to_avg": g.max() / g.mean(),
                         "night_share": df[df.daypart == "Night"].groupby("household")["kw"].sum() / g.sum()})


# ---------------------------------------------------------- 4. CLASSIFY
def add_level(df):
    q1, q2 = df["kw"].quantile([1 / 3, 2 / 3])
    df = df.copy()
    df["level"] = pd.cut(df["kw"], [-np.inf, q1, q2, np.inf], labels=["Low", "Medium", "High"])
    return df, (q1, q2)


def classify(df):
    df, cuts = add_level(df)
    d = df.copy()
    # lag features only use PAST values -> no target leakage
    d["lag_24"] = d["kw"].shift(24)
    d["lag_168"] = d["kw"].shift(168)
    d = d.dropna()
    feats = ["hour", "dow", "month", "weekend", "temp_c", "peak", "lag_24", "lag_168"]
    split = int(len(d) * 0.8)                     # time-based split
    tr, te = d.iloc[:split], d.iloc[split:]
    models = {"Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=SEED),
              "Random Forest": RandomForestClassifier(300, max_depth=12, n_jobs=-1, random_state=SEED)}
    res, best = [], None
    for name, m in models.items():
        m.fit(tr[feats], tr["level"])
        pred = m.predict(te[feats])
        res.append({"model": name, "accuracy": accuracy_score(te["level"], pred),
                    "macro_f1": f1_score(te["level"], pred, average="macro")})
        if name == "Random Forest":
            best = (m, pred)
    m, pred = best
    labels = ["Low", "Medium", "High"]
    return {"results": pd.DataFrame(res), "cuts": cuts,
            "confusion": pd.DataFrame(confusion_matrix(te["level"], pred, labels=labels), index=labels, columns=labels),
            "report": classification_report(te["level"], pred, output_dict=True),
            "importance": pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False)}


# ------------------------------------------------------- 5. ASSOCIATION
def association(df, min_support=0.03, min_conf=0.6, top=15):
    d, _ = add_level(df)
    t = pd.DataFrame({
        "Time": "Time=" + d["daypart"],
        "Day": np.where(d["weekend"] == 1, "Day=Weekend", "Day=Weekday"),
        "Season": "Season=" + d["season"],
        "Temp": "Temp=" + pd.cut(d["temp_c"], [-np.inf, 5, 15, 25, np.inf], labels=["Cold", "Cool", "Mild", "Hot"]).astype(str),
        "Use": "Use=" + d["level"].astype(str),
    })
    empty = pd.DataFrame(columns=["rule", "support", "confidence", "lift"])
    basket = pd.get_dummies(t.stack()).groupby(level=0).max().astype(bool)
    fi = apriori(basket, min_support=min_support, use_colnames=True)
    if fi.empty:
        return empty
    try:
        rules = association_rules(fi, metric="confidence", min_threshold=min_conf)
    except ValueError:
        return empty
    if rules.empty:
        return empty
    rules = rules[rules["consequents"].apply(lambda s: len(s) == 1 and next(iter(s)).startswith("Use="))]
    # keep rules that involve weather/season (drops the trivial "Evening -> High") and have >= 2 conditions
    rules = rules[rules["antecedents"].apply(lambda s: len(s) >= 2 and any(i.startswith(("Temp=", "Season=")) for i in s))]
    if rules.empty:
        return empty
    rules = rules[rules["lift"] > 1.05].sort_values(["lift", "confidence"], ascending=False)
    if rules.empty:
        return empty
    out = rules.assign(rule=lambda r: r["antecedents"].apply(lambda s: " & ".join(sorted(s))) + "  ->  " +
                       r["consequents"].apply(lambda s: ", ".join(s)))
    return out[["rule", "support", "confidence", "lift"]].head(top).reset_index(drop=True)


# ------------------------------------------------------------ 6. ANOMALY
def detect_anomalies(df, contamination=0.01):
    d = df.copy()
    base = d.groupby(["dow", "hour"])["kw"].transform("median")
    mad = d.groupby(["dow", "hour"])["kw"].transform(lambda x: (x - x.median()).abs().median()) + 1e-6
    d["robust_z"] = 0.6745 * (d["kw"] - base) / mad
    X = d[["kw", "hour", "dow", "temp_c"]].assign(resid=d["kw"] - base)
    iso = IsolationForest(contamination=contamination, random_state=SEED, n_jobs=-1).fit(X)
    d["iso_flag"] = iso.predict(X) == -1
    d["z_flag"] = d["robust_z"].abs() > 4
    d["anomaly"] = d["iso_flag"] & d["z_flag"]
    d["type"] = np.where(d["robust_z"] > 0, "Spike", "Drop")
    return d


# ------------------------------------------------------------- 7. INSIGHTS
def insights(df, patterns, clusters, anomalies):
    tot = df["kw"].sum()
    peak_share = df.loc[df.peak == 1, "kw"].sum() / tot * 100
    night = df.loc[df.daypart == "Night", "kw"]
    base_night = df.groupby("date")["kw"].min().median()
    wk = patterns["weekday_vs_weekend"]
    top_hours = patterns["hourly"].nlargest(3).index.tolist()
    seas = patterns["seasonal"]
    n_anom = int(anomalies["anomaly"].sum())
    out = [
        f"Peak window (17:00-21:59) accounts for {peak_share:.1f}% of consumption in only 5 of 24 hours; shifting flexible loads (washing, dishwasher, water heating) out of it cuts demand charges.",
        f"Highest-use hours on average: {', '.join(f'{h:02d}:00' for h in top_hours)}.",
        f"Typical daily minimum (standby baseline) is {base_night:.2f} kW, i.e. about {base_night * 24:.1f} kWh/day always-on load; check idle appliances.",
        f"{seas.idxmax()} is the highest-use season ({seas.max():.2f} kW avg), {seas.idxmin()} the lowest ({seas.min():.2f} kW): a {((seas.max() / seas.min()) - 1) * 100:.0f}% swing, mostly weather-driven.",
        f"Weekend daytime use (09-17h) is {(wk.loc[9:17, 'Weekend'].mean() / wk.loc[9:17, 'Weekday'].mean() - 1) * 100:+.0f}% vs weekdays.",
        f"{clusters['k']} daily load-profile clusters found; the highest-use cluster has {int((clusters['labels'] == clusters['centers'].mean(axis=1).idxmax()).sum())} days: target these days for demand-response tips.",
        f"{n_anom} anomalous hours flagged ({(anomalies.loc[anomalies.anomaly, 'type'] == 'Spike').sum()} spikes, {(anomalies.loc[anomalies.anomaly, 'type'] == 'Drop').sum()} drops): investigate spikes for faulty or left-on appliances, drops for outages/sensor faults.",
    ]
    return out


def run_all(raw=None, source=None):
    if raw is None:
        raw, source = load_data()
    df = preprocess(raw)
    patterns = temporal_patterns(df)
    prof = daily_profiles(df)
    clusters = cluster_days(prof)
    clf = classify(df)
    rules = association(df)
    anom = detect_anomalies(df)
    ins = insights(df, patterns, clusters, anom)
    return dict(source=source, df=df, patterns=patterns, profiles=prof, clusters=clusters,
                clf=clf, rules=rules, anomalies=anom, insights=ins)


def _in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _in_streamlit():  # wrong entry point: the dashboard is app.py
        import streamlit as st
        st.error("Run the dashboard with:  streamlit run app.py   (energy_mining.py is the console pipeline)")
        st.stop()
    r = run_all()
    print("Data source:", r["source"], "| rows:", len(r["df"]))
    print("\nClusters k =", r["clusters"]["k"]); print(r["clusters"]["scores"].round(3).to_string(index=False))
    print("\nClassification:"); print(r["clf"]["results"].round(3).to_string(index=False))
    print("Confusion matrix:\n", r["clf"]["confusion"])
    print("Top features:\n", r["clf"]["importance"].round(3).head(5))
    print("\nTop association rules:"); print(r["rules"].head(8).round(3).to_string(index=False))
    print("\nAnomalies flagged:", int(r["anomalies"]["anomaly"].sum()))
    print("\nInsights:")
    for ins in r["insights"]:
        print(" -", ins)
