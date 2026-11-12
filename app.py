"""Electricity Theft Detection - Streamlit front end.

Run:  streamlit run app.py
"""
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import model as M
from features import day_columns
from generate_data import generate

st.set_page_config(page_title="Electricity Theft Detection", page_icon="⚡", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "smart_meter_data.csv")


# ---------- bootstrapping ----------
@st.cache_resource
def get_bundle():
    if not os.path.exists(M.MODEL_PATH):
        os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
        df = generate()
        df.to_csv(DATA_PATH, index=False)
        M.train(df)
    return M.load()


@st.cache_data
def get_metrics():
    return M.load_metrics()


@st.cache_data
def load_default_data():
    if not os.path.exists(DATA_PATH):
        get_bundle()
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner="Scoring customers...")
def score_df(_df):
    return M.score(_df, get_bundle())


bundle = get_bundle()
metrics = get_metrics()

# ---------- sidebar ----------
st.sidebar.title("⚡ Theft Detection")
page = st.sidebar.radio("Navigate", ["Dashboard", "Customer Inspector", "Upload & Scan", "Model Performance", "About"])

st.sidebar.markdown("---")
src = st.sidebar.radio("Data source", ["Demo dataset (synthetic)", "Upload my CSV"])
data = None
if src == "Upload my CSV":
    up = st.sidebar.file_uploader("CSV: customer_id + d001..d365 daily kWh", type="csv")
    if up is not None:
        try:
            data = pd.read_csv(up)
            if len(day_columns(data)) < 300:
                st.sidebar.error("Need ~365 daily columns named d001, d002, ...")
                data = None
            elif "customer_id" not in data.columns:
                data.insert(0, "customer_id", [f"C{i+1:05d}" for i in range(len(data))])
        except Exception as e:  # noqa: BLE001
            st.sidebar.error(f"Could not read file: {e}")
if data is None:
    data = load_default_data()
    if src == "Upload my CSV":
        st.sidebar.info("Showing demo data until a valid file is uploaded.")

# trim/pad day columns to exactly 365 so features are consistent
dcols = day_columns(data)[:365]
if len(dcols) < 365:
    st.error("Dataset needs 365 daily columns.")
    st.stop()

scores = score_df(data)
view = data[["customer_id"] + dcols].copy()
merged = scores.copy()
if "is_theft" in data.columns:
    merged["actual_theft"] = data["is_theft"].values
    merged["theft_type"] = data.get("theft_type", pd.Series(["?"] * len(data))).values
merged["mean_kwh"] = data[dcols].mean(axis=1).values

# ---------- pages ----------
if page == "Dashboard":
    st.title("⚡ Electricity Theft Detection Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers scanned", f"{len(merged):,}")
    c2.metric("Flagged as suspicious", f"{int(merged.flagged.sum()):,}")
    c3.metric("Flag rate", f"{merged.flagged.mean():.1%}")
    c4.metric("High-risk customers", f"{int((merged.risk == 'High').sum()):,}")

    left, right = st.columns(2)
    with left:
        fig = px.histogram(merged, x="theft_probability", nbins=40, color="flagged",
                           color_discrete_map={True: "#e4572e", False: "#3a86ff"},
                           title="Distribution of theft probability")
        fig.add_vline(x=bundle["threshold"], line_dash="dash", annotation_text="decision threshold")
        st.plotly_chart(fig, width="stretch")
    with right:
        rc = merged["risk"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).reset_index()
        rc.columns = ["risk", "count"]
        fig = px.bar(rc, x="risk", y="count", color="risk", title="Customers by risk level",
                     color_discrete_map={"Low": "#2a9d8f", "Medium": "#f4a261", "High": "#e63946"})
        st.plotly_chart(fig, width="stretch")

    st.subheader("Top suspicious customers")
    n = st.slider("How many to show", 5, 100, 20)
    top = merged.sort_values("theft_probability", ascending=False).head(n)
    cols = ["customer_id", "theft_probability", "risk", "mean_kwh", "anomaly_score", "reasons"]
    if "actual_theft" in top:
        cols += ["actual_theft", "theft_type"]
    st.dataframe(top[cols].style.format({"theft_probability": "{:.1%}", "mean_kwh": "{:.1f}",
                                         "anomaly_score": "{:.3f}"}), width="stretch", height=420)
    st.download_button("⬇ Download full report (CSV)", merged.to_csv(index=False).encode(),
                       "theft_report.csv", "text/csv")

elif page == "Customer Inspector":
    st.title("🔍 Customer Inspector")
    default = merged.sort_values("theft_probability", ascending=False).customer_id.iloc[0]
    ids = merged.customer_id.tolist()
    cid = st.selectbox("Select customer", ids, index=ids.index(default))
    i = ids.index(cid)
    row = merged.iloc[i]
    series = data[dcols].iloc[i].to_numpy(dtype=float)

    c1, c2, c3 = st.columns(3)
    c1.metric("Theft probability", f"{row.theft_probability:.1%}")
    c2.metric("Risk level", str(row.risk))
    c3.metric("Verdict", "🚨 SUSPICIOUS" if row.flagged else "✅ Normal")

    days = np.arange(1, 366)
    roll = pd.Series(series).rolling(14, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=series, name="Daily kWh", opacity=0.45, line=dict(color="#3a86ff")))
    fig.add_trace(go.Scatter(x=days, y=roll, name="14-day average", line=dict(color="#e4572e", width=3)))
    fig.update_layout(title=f"Daily consumption — {cid}", xaxis_title="Day", yaxis_title="kWh", height=420)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Why was this customer scored this way?")
    for r in row.reasons.split(" | "):
        st.write(f"• {r}")
    if "actual_theft" in row:
        st.caption(f"Ground truth (demo data only): actual_theft = {row.actual_theft}, type = {row.theft_type}")

elif page == "Upload & Scan":
    st.title("📤 Scan a single customer's readings")
    st.write("Paste 365 daily kWh values (comma / space / newline separated), or generate a sample.")
    kind = st.selectbox("Generate sample", ["— none —", "Honest customer", "scale_down", "random_scale",
                                            "gradual_drop", "intermittent", "night_zero"])
    if st.button("Generate sample") and kind != "— none —":
        from generate_data import _apply_theft, _honest_profile
        rng = np.random.default_rng()
        s = _honest_profile(rng, 365)
        if kind != "Honest customer":
            s = _apply_theft(s, kind, rng)
        st.session_state["paste"] = ", ".join(f"{v:.2f}" for v in s)
    txt = st.text_area("Daily kWh values", st.session_state.get("paste", ""), height=160)
    if st.button("Analyze", type="primary"):
        try:
            vals = np.array([float(v) for v in txt.replace("\n", " ").replace(",", " ").split()])
            if len(vals) != 365:
                st.error(f"Expected 365 values, got {len(vals)}.")
            else:
                one = pd.DataFrame([vals], columns=[f"d{i+1:03d}" for i in range(365)])
                one.insert(0, "customer_id", "USER")
                res = M.score(one, bundle).iloc[0]
                a, b = st.columns(2)
                a.metric("Theft probability", f"{res.theft_probability:.1%}")
                b.metric("Verdict", "🚨 SUSPICIOUS" if res.flagged else "✅ Normal")
                st.plotly_chart(px.line(y=vals, labels={"x": "Day", "y": "kWh"}, title="Your series"),
                                width="stretch")
                for r in res.reasons.split(" | "):
                    st.write(f"• {r}")
        except ValueError:
            st.error("Could not parse values — use numbers only.")

elif page == "Model Performance":
    st.title("📊 Model Performance")
    t = metrics["test"]
    st.write(f"Selected model: **{metrics['best_model']}** (chosen by 5-fold CV PR-AUC) · "
             f"decision threshold **{metrics['threshold']:.2f}** (maximizes F1)")
    c = st.columns(5)
    c[0].metric("ROC-AUC", f"{t['roc_auc']:.3f}")
    c[1].metric("PR-AUC", f"{t['pr_auc']:.3f}")
    c[2].metric("Precision", f"{t['precision']:.3f}")
    c[3].metric("Recall", f"{t['recall']:.3f}")
    c[4].metric("F1", f"{t['f1']:.3f}")

    l, r = st.columns(2)
    with l:
        cm = np.array(t["confusion_matrix"])
        fig = px.imshow(cm, text_auto=True, x=["Pred normal", "Pred theft"], y=["Actual normal", "Actual theft"],
                        color_continuous_scale="Blues", title=f"Confusion matrix (hold-out, n={t['n_test']})")
        st.plotly_chart(fig, width="stretch")
    with r:
        fi = metrics.get("feature_importance")
        if fi:
            s = pd.Series(fi).sort_values().tail(12)
            fig_fi = px.bar(s, orientation="h", title="Top features", labels={"value": "importance", "index": ""})
            fig_fi.update_layout(showlegend=False)
            st.plotly_chart(fig_fi, width="stretch")
    st.subheader("Model comparison (cross-validation)")
    st.dataframe(pd.DataFrame(metrics["candidates"]).T.style.format("{:.3f}"), width="stretch")
    if st.button("🔁 Retrain on current data"):
        if "is_theft" in data.columns:
            with st.spinner("Training..."):
                M.train(data)
            st.cache_resource.clear()
            st.cache_data.clear()
            st.success("Retrained. Reload the page.")
        else:
            st.error("Current data has no `is_theft` label column.")

else:
    st.title("ℹ️ About")
    st.markdown("""
**What it does:** Flags electricity customers whose smart-meter readings look like theft
(meter tampering, bypass, under-reporting).

**Pipeline**
1. Daily kWh series per customer (365 days) →
2. 20 engineered features (level, volatility, trend, seasonality fit, low/zero-day runs, peer comparison) →
3. Supervised classifier (best of Logistic Regression / Random Forest / Gradient Boosting) +
   Isolation Forest anomaly score →
4. Probability, risk band, and rule-based plain-English reasons.

**Tech stack:** Python · Streamlit · pandas · NumPy · scikit-learn · Plotly · joblib

**Dataset:** Synthetic smart-meter data generated by `generate_data.py` (1,200 customers × 365 days,
12% theft) using 5 attack patterns from the electricity-theft literature. Swap in real data
(e.g. the SGCC dataset) by uploading a CSV with `customer_id`, `d001`…`d365` (and `is_theft` to retrain).
""")
