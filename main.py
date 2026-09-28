"""Run the full pipeline: data -> KPIs -> charts -> churn model."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_curve

from src import churn_model as cm
from src import kpi_metrics as km
from src.data_generator import END, PLANS, load_data

os.makedirs("outputs", exist_ok=True)
sns.set_theme(style="whitegrid")


def main():
    d = load_data("data")
    subs, ws, rev = d["subscribers"], d["watch_sessions"], d["revenue"]
    k = km.compute_kpis(subs, ws, rev, END)
    print("=" * 50)
    print(f"  KPI scorecard ({k['month']})")
    print("=" * 50)
    print(f"  MAU {k['mau']:,} | Avg DAU {k['dau']:,.0f} | Monthly ARPU Rs {k['arpu']:,.0f}")
    print(f"  Monthly churn {k['monthly_churn']:.2f}% | Lifetime churn {k['lifetime_churn']:.1f}%")
    print(f"  Completion {k['completion']:.1f}% | Hours {k['hours']:,.0f} | Revenue Rs {k['revenue']:,.0f}")

    t = km.monthly_trend(ws, rev)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for a, c in zip(ax, ["sessions", "viewers", "hours"]):
        a.plot(t["month"], t[c], marker="o"); a.set_title(c.title())
        a.set_xticks(t["month"][::3]); a.tick_params(axis="x", rotation=45)
    plt.tight_layout(); plt.savefig("outputs/viewership_trends.png", dpi=120); plt.close()

    g = km.genre_table(ws, d["content_catalog"], rev)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for a, c in zip(ax, ["views", "completion", "attributed_revenue"]):
        sns.barplot(x="genre", y=c, data=g, ax=a); a.set_title(c); a.tick_params(axis="x", rotation=45)
    plt.tight_layout(); plt.savefig("outputs/genre_performance.png", dpi=120); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.barplot(x="plan_type", y="churned", data=subs, order=PLANS, ax=ax[0])
    sns.barplot(x="region", y="churned", data=subs, ax=ax[1])
    plt.tight_layout(); plt.savefig("outputs/churn_analysis.png", dpi=120); plt.close()

    plt.figure(figsize=(14, 4)); sns.heatmap(km.weekday_hour(ws), cmap="YlOrRd")
    plt.tight_layout(); plt.savefig("outputs/viewing_heatmap.png", dpi=120); plt.close()

    print("\n[Churn] Training model...")
    res = cm.train(d)
    thr = cm.threshold_for_recall(res, 0.70)
    print(f"  Test ROC-AUC {res['auc']:.3f} | 5-fold CV AUC {res['cv_mean']:.3f} +/- {res['cv_std']:.3f}")
    print(f"  Threshold for ~70% recall: {thr:.3f}")
    print(classification_report(res["y_te"], (res["proba"] >= thr).astype(int),
                                target_names=["Retained", "Churned"], digits=3))
    fpr, tpr, _ = roc_curve(res["y_te"], res["proba"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "--", color="grey"); ax[0].set_title("ROC curve")
    res["importance"].plot.barh(ax=ax[1]); ax[1].set_title("Feature importance")
    plt.tight_layout(); plt.savefig("outputs/churn_model_analysis.png", dpi=120); plt.close()
    print("Done. Charts saved to outputs/")


if __name__ == "__main__":
    main()
