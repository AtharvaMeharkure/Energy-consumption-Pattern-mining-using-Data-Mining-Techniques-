"""Run with:  streamlit run app.py"""
import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import energy_mining as em

st.set_page_config(page_title="Energy Consumption Pattern Mining", page_icon="⚡", layout="wide")


@st.cache_data(show_spinner="Running data mining pipeline...")
def get_results(data=None):
    if data is None:
        return em.run_all()
    raw, source = em.load_csv(io.BytesIO(data))
    return em.run_all(raw, source)


with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Upload CSV", type="csv",
                          help="Columns: datetime (or Date + Time) and kw. Optional: temp_c. At least 30 days. Any interval.")
    st.caption("No file: built-in dataset (UCI file if present, else synthetic).")
try:
    R = get_results(up.getvalue() if up else None)
except ValueError as e:
    st.error(str(e))
    st.stop()
df, P, C, K, A = R["df"], R["patterns"], R["clusters"], R["clf"], R["anomalies"]

st.title("⚡ Energy Consumption Pattern Mining")
st.caption(f"Data source: **{R['source']}** · {len(df):,} hourly readings · "
           f"{df.index.min():%d %b %Y} to {df.index.max():%d %b %Y}")

# ---- sidebar filters
with st.sidebar:
    st.divider()
    st.header("Filters")
    dmin, dmax = df.index.min().date(), df.index.max().date()
    rng = st.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
    seasons = st.multiselect("Season", ["Winter", "Spring", "Summer", "Autumn"], default=["Winter", "Spring", "Summer", "Autumn"])
f = df
if isinstance(rng, tuple) and len(rng) == 2:
    f = f[(f.index.date >= rng[0]) & (f.index.date <= rng[1])]
f = f[f["season"].isin(seasons)]
if f.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# ---- KPIs
c = st.columns(5)
c[0].metric("Avg load", f"{f.kw.mean():.2f} kW")
c[1].metric("Peak load", f"{f.kw.max():.2f} kW")
c[2].metric("Total energy", f"{f.kw.sum():,.0f} kWh")
c[3].metric("Peak-window share", f"{f.loc[f.peak == 1, 'kw'].sum() / f.kw.sum() * 100:.1f}%")
c[4].metric("Anomalies", int(A.loc[f.index.intersection(A.index), "anomaly"].sum()))

t1, t2, t3, t4, t5, t6 = st.tabs(["Patterns", "Clusters", "Classification", "Association rules", "Anomalies", "Insights"])

with t1:
    hp = f.pivot_table(index="dayname", columns="hour", values="kw", aggfunc="mean").reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    st.plotly_chart(px.imshow(hp, aspect="auto", color_continuous_scale="YlOrRd", labels=dict(x="Hour", y="", color="kW"),
                              title="Average load: day of week × hour"), width="stretch")
    a, b = st.columns(2)
    wk = f.groupby(["weekend", "hour"])["kw"].mean().unstack(0).rename(columns={0: "Weekday", 1: "Weekend"})
    a.plotly_chart(px.line(wk, title="Weekday vs weekend profile", labels=dict(value="kW", hour="Hour")), width="stretch")
    m = f.groupby("month")["kw"].mean()
    b.plotly_chart(px.bar(m, title="Monthly average load", labels=dict(value="kW", month="Month")).update_layout(showlegend=False),
                   width="stretch")
    st.plotly_chart(px.scatter(f.sample(min(4000, len(f)), random_state=1), x="temp_c", y="kw", color="daypart", opacity=.5,
                               title="Load vs temperature", labels=dict(temp_c="Temperature (°C)", kw="kW")), width="stretch")
    if "proxy" in R["source"]:
        st.info("Temperature here is a seasonal proxy, not measured weather. Add weather.csv (datetime,temp_c) for real weather.")

with t2:
    st.write(f"Daily 24-hour load profiles clustered with K-Means (**k = {C['k']}**, chosen by silhouette score, k ≥ 3).")
    a, b = st.columns(2)
    cen = C["centers"].T
    a.plotly_chart(px.line(cen, title="Cluster centroids (average daily profile)", labels=dict(value="kW", index="Hour", variable="")),
                   width="stretch")
    pc = C["pca"].assign(cluster=C["labels"])
    b.plotly_chart(px.scatter(pc, x="PC1", y="PC2", color="cluster", title="Days in PCA space"), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(px.line(C["scores"], x="k", y="inertia", markers=True, title="Elbow curve"), width="stretch")
    b.plotly_chart(px.line(C["scores"], x="k", y="silhouette", markers=True, title="Silhouette score"), width="stretch")
    dist = C["labels"].value_counts().rename_axis("cluster").reset_index(name="days")
    st.dataframe(dist, hide_index=True)

with t3:
    q1, q2 = K["cuts"]
    st.write(f"Consumption level: **Low** ≤ {q1:.2f} kW < **Medium** ≤ {q2:.2f} kW < **High** (tertiles). "
             "Time-based 80/20 split; lag features use past values only (no leakage).")
    st.dataframe(K["results"].round(3), hide_index=True)
    a, b = st.columns(2)
    a.plotly_chart(px.imshow(K["confusion"], text_auto=True, color_continuous_scale="Blues", title="Confusion matrix (Random Forest)",
                             labels=dict(x="Predicted", y="Actual")), width="stretch")
    b.plotly_chart(px.bar(K["importance"].sort_values(), orientation="h", title="Feature importance").update_layout(showlegend=False),
                   width="stretch")

with t4:
    st.write("Apriori rules with weather/season in the condition and consumption level as the outcome. "
             "**Lift > 1** means the condition makes that outcome more likely than average.")
    if R["rules"].empty:
        st.info("No rules passed the thresholds (support 0.03, confidence 0.6, lift > 1.05). "
                "This happens when the data has little variety in season or temperature, e.g. a short file covering one season.")
    else:
        st.dataframe(R["rules"].round(3), hide_index=True, width="stretch")

with t5:
    fa = A.loc[A.index.isin(f.index)]
    an = fa[fa["anomaly"]]
    fig = go.Figure()
    fig.add_scatter(x=fa.index, y=fa["kw"], mode="lines", name="kW", line=dict(width=1))
    for t, col in [("Spike", "red"), ("Drop", "orange")]:
        s = an[an["type"] == t]
        fig.add_scatter(x=s.index, y=s["kw"], mode="markers", name=t, marker=dict(color=col, size=8))
    fig.update_layout(title="Load with detected anomalies (Isolation Forest + robust z-score)")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(an[["kw", "robust_z", "type", "hour", "dayname"]].sort_values("robust_z", key=abs, ascending=False).round(2).head(50),
                 width="stretch")

with t6:
    st.subheader("Energy-saving insights")
    for i in R["insights"]:
        st.markdown(f"- {i}")
