"""OTT Media Analytics: interactive Streamlit dashboard.  Run:  streamlit run app.py"""
import pandas as pd
import plotly.express as px
import streamlit as st

from src import churn_model as cm
from src import kpi_metrics as km
from src.data_generator import END, PLANS, PRICE, START, load_data

st.set_page_config(page_title="OTT Media Analytics", page_icon="📺", layout="wide")


@st.cache_data(show_spinner="Loading data (first run generates it)...")
def get_data():
    return load_data("data")


@st.cache_resource(show_spinner="Training churn model...")
def get_model():
    return cm.train(get_data())


d = get_data()
content = d["content_catalog"]

# ---------------- sidebar filters ----------------
st.sidebar.title("📺 Filters")
plans = st.sidebar.multiselect("Plan", PLANS, default=PLANS)
regions = st.sidebar.multiselect("Region", sorted(d["subscribers"]["region"].unique()),
                                 default=sorted(d["subscribers"]["region"].unique()))
devices = st.sidebar.multiselect("Device", sorted(d["subscribers"]["device_type"].unique()),
                                 default=sorted(d["subscribers"]["device_type"].unique()))
dates = st.sidebar.date_input("Date range", (START.date(), END.date()),
                              min_value=START.date(), max_value=END.date())
if len(dates) != 2:
    st.info("Select both a start and an end date.")
    st.stop()
start, end = dates
subs, ws, rev = km.filter_data(d, plans, regions, devices, start, end)
if subs.empty or ws.empty:
    st.warning("No data for these filters.")
    st.stop()

st.title("OTT Media Analytics")
st.caption("Synthetic data | KPIs are computed for the last month of the selected date range")

k = km.compute_kpis(subs, ws, rev, end)
c = st.columns(4)
c[0].metric("MAU", f"{k['mau']:,}")
c[1].metric("Avg DAU (30d)", f"{k['dau']:,.0f}")
c[2].metric("Monthly ARPU", f"₹{k['arpu']:,.0f}")
c[3].metric("Monthly churn", f"{k['monthly_churn']:.2f}%")
c = st.columns(4)
c[0].metric("Lifetime churn", f"{k['lifetime_churn']:.1f}%")
c[1].metric("Completion rate", f"{k['completion']:.1f}%")
c[2].metric("Watch hours", f"{k['hours']:,.0f}")
c[3].metric("Revenue (period)", f"₹{k['revenue'] / 1e6:,.1f}M")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Churn & cohorts", "Content ROI", "Churn model", "Data"])

# ---------------- overview ----------------
with tab1:
    t = km.monthly_trend(ws, rev)
    metric = st.radio("Trend metric", ["sessions", "viewers", "hours", "revenue", "paying_subs"],
                      horizontal=True)
    st.plotly_chart(px.line(t, x="month", y=metric, markers=True))
    g = km.genre_table(ws, content, rev)
    left, right = st.columns(2)
    gm = left.selectbox("Genre metric", ["views", "completion", "hours", "attributed_revenue"])
    left.plotly_chart(px.bar(g, x="genre", y=gm))
    right.subheader("Sessions by weekday and hour")
    right.plotly_chart(px.imshow(km.weekday_hour(ws), aspect="auto", color_continuous_scale="YlOrRd"))
    dv = ws.merge(subs[["subscriber_id", "device_type"]], on="subscriber_id").groupby("device_type").agg(
        sessions=("session_id", "count"), completion=("completion_pct", "mean")).reset_index()
    st.subheader("Completion rate by device")
    st.plotly_chart(px.bar(dv, x="device_type", y="completion"))

# ---------------- churn & cohorts ----------------
with tab2:
    a, b = st.columns(2)
    a.subheader("Lifetime churn by plan (%)")
    a.plotly_chart(px.bar(subs.groupby("plan_type")["churned"].mean().mul(100).reset_index(),
                          x="plan_type", y="churned"))
    b.subheader("Lifetime churn by region (%)")
    b.plotly_chart(px.bar(subs.groupby("region")["churned"].mean().mul(100).reset_index(),
                          x="region", y="churned"))
    st.subheader("Cohort retention (% of join-month cohort still paying, by months since joining)")
    coh = km.cohort_retention(subs, rev)
    st.plotly_chart(px.imshow(coh, aspect="auto", color_continuous_scale="Blues", text_auto=".0f",
                              labels=dict(x="Months since joining", y="Join month", color="% retained")))

# ---------------- content ROI ----------------
with tab3:
    roi = km.content_roi(ws, content, rev)
    st.caption("Revenue is attributed to titles by each subscriber's watch-time share. "
               "ROI = (attributed revenue − production cost) / production cost.")
    gsel = st.multiselect("Genres", sorted(roi["genre"].unique()), default=sorted(roi["genre"].unique()))
    r = roi[roi["genre"].isin(gsel)]
    st.plotly_chart(px.scatter(r, x="production_cost", y="attributed_revenue", color="genre", size="views",
                               hover_name="title", log_x=True, log_y=True))
    cols = ["title", "genre", "views", "completion", "production_cost", "attributed_revenue", "roi_pct"]
    x, y = st.columns(2)
    x.subheader("Best ROI")
    x.dataframe(r.nlargest(10, "roi_pct")[cols].round(1), hide_index=True)
    y.subheader("Worst ROI (candidates to cut)")
    y.dataframe(r.nsmallest(10, "roi_pct")[cols].round(1), hide_index=True)

# ---------------- churn model ----------------
with tab4:
    res = get_model()
    st.write(f"Random Forest predicting churn in the next 180 days from a 1 Jul 2025 snapshot. "
             f"**Test ROC-AUC {res['auc']:.3f}**, 5-fold CV AUC {res['cv_mean']:.3f} ± {res['cv_std']:.3f}. "
             "Features use only pre-snapshot data (no leakage).")
    thr = st.slider("Decision threshold", 0.05, 0.9, round(cm.threshold_for_recall(res, 0.70), 2), 0.01)
    s = cm.threshold_stats(res, thr)
    m = st.columns(3)
    m[0].metric("Recall", f"{s['recall']:.0%}")
    m[1].metric("Precision", f"{s['precision']:.0%}")
    m[2].metric("Flagged subscribers", f"{s['flagged']:,}")
    a, b = st.columns(2)
    a.subheader("Feature importance")
    a.plotly_chart(px.bar(res["importance"].reset_index(), x=0, y="index", orientation="h"))
    b.subheader("Top at-risk subscribers")
    risk = res["risk"].reset_index().merge(subs[["subscriber_id", "plan_type", "region"]], on="subscriber_id")
    risk = risk.merge(res["X"][["recency_days", "sessions_last_30d"]].reset_index(), on="subscriber_id")
    b.dataframe(risk.sort_values("churn_risk", ascending=False).head(15).round(2), hide_index=True)
    st.subheader("Retention campaign simulator")
    z = st.columns(3)
    save = z[0].slider("Save rate of true churners", 5, 60, 25) / 100
    cost = z[1].slider("Offer cost per flagged subscriber (₹)", 0, 300, 50, 10)
    months = z[2].slider("Months of revenue saved", 1, 12, 6)
    arpu_pop = float(subs["plan_type"].map(PRICE).mean())
    true_churners = s["flagged"] * s["precision"]
    net = true_churners * save * arpu_pop * months - s["flagged"] * cost
    st.metric("Net impact of campaign", f"₹{net:,.0f}", delta="profit" if net >= 0 else "loss")

# ---------------- data ----------------
with tab5:
    table = st.selectbox("Table", list(d.keys()))
    st.dataframe(d[table].head(1000), hide_index=True)
    st.download_button("Download filtered sessions (CSV)", ws.to_csv(index=False), "watch_sessions_filtered.csv")
