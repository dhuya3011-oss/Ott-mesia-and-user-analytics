import os
import numpy as np
import pandas as pd

np.random.seed(42)

START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2025-12-31")

PLANS = ["Basic", "Standard", "Premium"]
PRICE = {"Basic": 149, "Standard": 499, "Premium": 649}
GENRES = ["Drama", "Thriller", "Comedy", "Action", "Romance", "Documentary", "Sci-Fi", "Horror"]
FILES = ["content_catalog", "subscribers", "watch_sessions", "revenue"]


def make_content(n=500):
    df = pd.DataFrame()
    df["content_id"] = range(1, n + 1)
    df["title"] = ["Title " + str(i).zfill(3) for i in range(1, n + 1)]
    df["genre"] = np.random.choice(GENRES, n, p=[0.20, 0.16, 0.14, 0.14, 0.10, 0.08, 0.10, 0.08])
    df["content_type"] = np.random.choice(["Movie", "Series", "Documentary"], n, p=[0.45, 0.45, 0.10])
    df["release_year"] = np.random.randint(2005, 2025, n)
    df["duration_mins"] = np.random.randint(25, 160, n)
    df["language"] = np.random.choice(["Hindi", "English", "Tamil", "Telugu", "Marathi"], n, p=[0.4, 0.3, 0.1, 0.1, 0.1])
    df["production_cost"] = np.random.randint(10, 200, n) * 1000
    return df


def make_subscribers(n=10000):
    df = pd.DataFrame()
    df["subscriber_id"] = range(1, n + 1)
    df["region"] = np.random.choice(["West", "North", "South", "East", "Central"], n)
    df["device_type"] = np.random.choice(["Mobile", "Smart TV", "Laptop", "Tablet"], n, p=[0.5, 0.25, 0.15, 0.1])
    df["plan_type"] = np.random.choice(PLANS, n, p=[0.45, 0.35, 0.20])
    df["join_date"] = START + pd.to_timedelta(np.random.randint(0, 547, n), unit="D")

    engagement = np.random.gamma(2.0, 1.0, n)

    base = df["plan_type"].map({"Basic": 0.30, "Standard": 0.20, "Premium": 0.12}).values
    chance = base * (1.7 - 0.35 * np.minimum(engagement, 3))
    chance = np.clip(chance, 0.02, 0.85)
    churns = np.random.random(n) < chance

    churn_date = df["join_date"] + pd.to_timedelta(np.random.randint(45, 400, n), unit="D")
    df["churn_date"] = churn_date.where(churns & (churn_date <= END))
    df["churned"] = df["churn_date"].notna().astype(int)
    df["monthly_revenue"] = df["plan_type"].map(PRICE)
    return df, engagement


def make_sessions(subs, engagement, content):
    n = len(subs)

    boost = content["genre"].map({"Drama": 1.8, "Thriller": 1.6}).fillna(1.0).values
    popularity = (np.random.pareto(2.5, len(content)) + 1) * boost
    popularity = popularity / popularity.sum()

    last_day = subs["churn_date"].fillna(END)
    days_active = (last_day - subs["join_date"]).dt.days.values
    days_active = np.maximum(days_active, 1)

    sessions_each = np.random.poisson(engagement * days_active / 30)
    who = np.repeat(np.arange(n), sessions_each)

    r = np.random.random(len(who))
    leaving = subs["churned"].values[who] == 1
    day_offset = days_active[who] * np.where(leaving, r ** 1.5, r)

    hour_weights = np.array([1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 4, 5, 5, 5, 5, 6, 7, 9, 12, 14, 14, 10, 4], dtype=float)
    hour_weights = hour_weights / hour_weights.sum()

    ws = pd.DataFrame()
    ws["subscriber_id"] = subs["subscriber_id"].values[who]
    ws["content_id"] = np.random.choice(content["content_id"].values, len(who), p=popularity)
    ws["watch_date"] = subs["join_date"].values[who] + pd.to_timedelta(day_offset.astype(int), unit="D")
    ws["watch_date"] = pd.to_datetime(ws["watch_date"]).dt.normalize()
    ws["watch_hour"] = np.random.choice(24, len(who), p=hour_weights)
    ws["completion_pct"] = np.clip(np.random.normal(55, 25, len(who)), 1, 100).round(1)

    ws = ws[ws["watch_date"] <= END].reset_index(drop=True)
    ws.insert(0, "session_id", range(1, len(ws) + 1))

    lengths = content.set_index("content_id")["duration_mins"]
    ws["watch_mins"] = (ws["content_id"].map(lengths) * ws["completion_pct"] / 100).round(1)
    return ws


def make_revenue(subs):
    rows = []
    for i in range(len(subs)):
        sid = subs["subscriber_id"].iloc[i]
        plan = subs["plan_type"].iloc[i]
        start = subs["join_date"].iloc[i]
        stop = subs["churn_date"].iloc[i]
        if pd.isna(stop):
            stop = END
        for month in pd.period_range(start, stop, freq="M"):
            rows.append((sid, str(month), PRICE[plan], plan))

    rev = pd.DataFrame(rows, columns=["subscriber_id", "month", "amount", "plan_type"])
    rev.insert(0, "revenue_id", range(1, len(rev) + 1))
    return rev


def generate():
    np.random.seed(42)
    content = make_content()
    subs, engagement = make_subscribers()
    ws = make_sessions(subs, engagement, content)
    rev = make_revenue(subs)
    return {"content_catalog": content, "subscribers": subs, "watch_sessions": ws, "revenue": rev}


def save_all(data, data_dir="data"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for name in data:
        data[name].to_csv(data_dir + "/" + name + ".csv", index=False)


def data_is_valid(data_dir="data"):
    for name in FILES:
        path = data_dir + "/" + name + ".csv"
        if not os.path.exists(path) or os.path.getsize(path) < 100:
            return False
    return True


def load_data(data_dir="data"):
    if not data_is_valid(data_dir):
        print("[Data] making datasets...")
        save_all(generate(), data_dir)

    d = {}
    for name in FILES:
        d[name] = pd.read_csv(data_dir + "/" + name + ".csv")

    d["subscribers"]["join_date"] = pd.to_datetime(d["subscribers"]["join_date"])
    d["subscribers"]["churn_date"] = pd.to_datetime(d["subscribers"]["churn_date"])
    d["watch_sessions"]["watch_date"] = pd.to_datetime(d["watch_sessions"]["watch_date"])
    return d
