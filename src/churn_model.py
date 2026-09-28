"""Churn model: features come only from data before a snapshot date (no leakage);
the label is 'churns within the next 180 days'."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, train_test_split

SNAP = pd.Timestamp("2025-07-01")
HORIZON_DAYS = 180


def build_features(d, snap=SNAP, horizon_days=HORIZON_DAYS):
    subs, ws, rev = d["subscribers"], d["watch_sessions"], d["revenue"]
    pop = subs[(subs["join_date"] < snap) & (subs["churn_date"].isna() | (subs["churn_date"] >= snap))].copy()
    pop["label"] = ((pop["churn_date"] >= snap) &
                    (pop["churn_date"] < snap + pd.Timedelta(days=horizon_days))).astype(int)
    past = ws[ws["watch_date"] < snap]
    agg = past.groupby("subscriber_id").agg(total_sessions=("session_id", "count"),
                                            avg_completion=("completion_pct", "mean"),
                                            avg_watch_mins=("watch_mins", "mean"),
                                            last_watch=("watch_date", "max"))
    recent = (past[past["watch_date"] >= snap - pd.Timedelta(days=30)]
              .groupby("subscriber_id").size().rename("sessions_last_30d"))
    pay = rev[rev["month"] < snap.strftime("%Y-%m")].groupby("subscriber_id")["amount"].sum().rename("total_revenue")
    X = pop.set_index("subscriber_id")[["plan_type", "join_date", "label"]].join([agg, recent, pay])
    X["tenure_days"] = (snap - X["join_date"]).dt.days
    X["recency_days"] = (snap - X["last_watch"]).dt.days.fillna(X["tenure_days"])
    X["is_premium"] = (X["plan_type"] == "Premium").astype(int)
    X["is_standard"] = (X["plan_type"] == "Standard").astype(int)
    X = X.drop(columns=["plan_type", "join_date", "last_watch"]).fillna(0)
    y = X.pop("label")
    return X, y


def train(d, seed=42):
    X, y = build_features(d)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                   random_state=seed, n_jobs=-1).fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    cv = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    return {"model": model, "X": X, "y": y, "y_te": y_te.values, "proba": proba,
            "auc": roc_auc_score(y_te, proba), "cv_mean": cv.mean(), "cv_std": cv.std(),
            "risk": pd.Series(model.predict_proba(X)[:, 1], index=X.index, name="churn_risk"),
            "importance": pd.Series(model.feature_importances_, index=X.columns).sort_values()}


def threshold_stats(res, threshold):
    """Precision / recall on the held-out test set at a chosen threshold."""
    pred = res["proba"] >= threshold
    tp = int((pred & (res["y_te"] == 1)).sum())
    precision = tp / pred.sum() if pred.sum() else 0.0
    recall = tp / max(int(res["y_te"].sum()), 1)
    flagged = int((res["risk"] >= threshold).sum())
    return {"precision": precision, "recall": recall, "flagged": flagged}


def threshold_for_recall(res, target=0.70):
    fpr, tpr, thr = roc_curve(res["y_te"], res["proba"])
    return float(thr[np.argmax(tpr >= target)])
