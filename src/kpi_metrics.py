"""KPI engine and reusable aggregations (used by main.py and the Streamlit app)."""
import numpy as np
import pandas as pd


def filter_data(d, plans=None, regions=None, devices=None, start=None, end=None):
    subs = d["subscribers"]
    if plans: subs = subs[subs["plan_type"].isin(plans)]
    if regions: subs = subs[subs["region"].isin(regions)]
    if devices: subs = subs[subs["device_type"].isin(devices)]
    ids = subs["subscriber_id"]
    ws = d["watch_sessions"][d["watch_sessions"]["subscriber_id"].isin(ids)]
    rev = d["revenue"][d["revenue"]["subscriber_id"].isin(ids)]
    if start is not None: ws = ws[ws["watch_date"] >= pd.Timestamp(start)]
    if end is not None: ws = ws[ws["watch_date"] <= pd.Timestamp(end)]
    if start is not None: rev = rev[rev["month"] >= pd.Timestamp(start).strftime("%Y-%m")]
    if end is not None: rev = rev[rev["month"] <= pd.Timestamp(end).strftime("%Y-%m")]
    return subs, ws, rev


def compute_kpis(subs, ws, rev, end):
    end = pd.Timestamp(end)
    month = end.strftime("%Y-%m")
    prev = (end.to_period("M") - 1).strftime("%Y-%m")
    mau = ws.loc[ws["watch_date"].dt.strftime("%Y-%m") == month, "subscriber_id"].nunique()
    last30 = ws[ws["watch_date"] > end - pd.Timedelta(days=30)]
    dau = last30.groupby("watch_date")["subscriber_id"].nunique().mean() if len(last30) else 0
    rm = rev[rev["month"] == month]
    arpu = rm["amount"].sum() / rm["subscriber_id"].nunique() if len(rm) else 0
    active_prev = rev[rev["month"] == prev]["subscriber_id"].nunique()
    churned = (subs["churn_date"].dt.strftime("%Y-%m") == month).sum()
    return {
        "month": month, "mau": int(mau), "dau": float(dau), "arpu": float(arpu),
        "monthly_churn": 100 * churned / active_prev if active_prev else 0.0,
        "lifetime_churn": 100 * subs["churned"].mean() if len(subs) else 0.0,
        "completion": float(ws["completion_pct"].mean()) if len(ws) else 0.0,
        "hours": float(ws["watch_mins"].sum() / 60), "revenue": float(rev["amount"].sum()),
        "subscribers": int(len(subs)),
    }


def monthly_trend(ws, rev):
    ws = ws.assign(month=ws["watch_date"].dt.strftime("%Y-%m"))
    m = ws.groupby("month").agg(sessions=("session_id", "count"), viewers=("subscriber_id", "nunique"),
                                hours=("watch_mins", lambda s: s.sum() / 60))
    r = rev.groupby("month").agg(revenue=("amount", "sum"), paying_subs=("subscriber_id", "nunique"))
    return m.join(r, how="outer").fillna(0).reset_index()


def genre_table(ws, content, rev):
    """Genre views/completion/hours plus revenue split by each subscriber's watch-time share."""
    w = ws.merge(content[["content_id", "genre"]], on="content_id")
    g = w.groupby("genre").agg(views=("session_id", "count"), completion=("completion_pct", "mean"),
                               hours=("watch_mins", lambda s: s.sum() / 60))
    sub_rev = rev.groupby("subscriber_id")["amount"].sum()
    sg = w.groupby(["subscriber_id", "genre"])["watch_mins"].sum().reset_index()
    sg["tot"] = sg.groupby("subscriber_id")["watch_mins"].transform("sum")
    sg["attr"] = sg["subscriber_id"].map(sub_rev).fillna(0) * sg["watch_mins"] / sg["tot"]
    g["attributed_revenue"] = sg.groupby("genre")["attr"].sum()
    return g.reset_index().sort_values("views", ascending=False)


def content_roi(ws, content, rev):
    """Revenue attributed to each title (by watch-time share) vs its production cost."""
    sub_rev = rev.groupby("subscriber_id")["amount"].sum()
    w = ws[["subscriber_id", "content_id", "watch_mins", "completion_pct", "session_id"]].copy()
    w["share"] = w["watch_mins"] / w.groupby("subscriber_id")["watch_mins"].transform("sum")
    w["attr"] = w["subscriber_id"].map(sub_rev).fillna(0) * w["share"]
    t = w.groupby("content_id").agg(views=("session_id", "count"), completion=("completion_pct", "mean"),
                                    attributed_revenue=("attr", "sum")).reset_index()
    t = content[["content_id", "title", "genre", "production_cost"]].merge(t, on="content_id", how="left").fillna(0)
    t["engagement_score"] = t["views"] * t["completion"] / 100
    t["roi_pct"] = 100 * (t["attributed_revenue"] - t["production_cost"]) / t["production_cost"]
    return t


def weekday_hour(ws):
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    x = ws.assign(weekday=ws["watch_date"].dt.day_name())
    return (x.pivot_table(index="weekday", columns="watch_hour", values="session_id", aggfunc="count")
             .reindex(order).reindex(columns=range(24)).fillna(0))


def cohort_retention(subs, rev):
    """Share of each join-month cohort still paying N months after joining."""
    s = subs[["subscriber_id", "join_date"]].copy()
    s["cohort"] = s["join_date"].dt.to_period("M")
    r = rev[["subscriber_id", "month"]].merge(s, on="subscriber_id")
    r["age"] = (pd.PeriodIndex(r["month"], freq="M") - r["cohort"]).map(lambda x: x.n)
    size = s.groupby("cohort").size()
    piv = r.groupby(["cohort", "age"])["subscriber_id"].nunique().unstack(fill_value=0)
    piv = piv.div(size, axis=0) * 100
    piv.index = piv.index.astype(str)
    return piv[[c for c in piv.columns if c <= 12]].round(1)
