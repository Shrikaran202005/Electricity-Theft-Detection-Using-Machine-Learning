"""Report-aligned experiment: reproduces the pipeline described in the project report.

Design (matches the TEAM-8 report):
  * 5,000 customer profiles x 180 daily readings, 8% theft
  * Features: avg/std daily kWh, max spike, zero-reading days, consumption-drop ratio, CV
  * Stratified 80/20 split, SMOTE on the training split only
  * Supervised: Random Forest + XGBoost    Unsupervised: Isolation Forest + One-Class SVM
  * Outputs figures 4.1-5.3 into figures/ and numbers into figures/report_metrics.json

Run:  python report_pipeline.py
NOTE: the dataset is synthetic (generate_data.py), so the numbers are produced by this code
and will differ from any figures quoted from a different dataset.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier

from generate_data import generate

N_CUSTOMERS, N_DAYS, THEFT_RATIO, SEED = 5000, 180, 0.08, 42
FIG_DIR = "figures"
sns.set_theme(style="whitegrid", context="talk")
PAL = {"normal": "#2a9d8f", "theft": "#e63946", "blue": "#264653"}


def report_features(df):
    x = df[[c for c in df.columns if c.startswith("d") and c[1:].isdigit()]].to_numpy(float)
    avg = x.mean(1)
    f = pd.DataFrame({
        "avg_daily_kwh": avg,
        "std_daily_kwh": x.std(1),
        "max_kwh_spike": x.max(1),
        "zero_reading_days": (x < 0.05 * avg[:, None]).sum(1),
        "consumption_drop_ratio": x[:, -30:].mean(1) / (x[:, :-30].mean(1) + 1e-9),
        "cv": x.std(1) / (avg + 1e-9),
    })
    return f


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = generate(N_CUSTOMERS, THEFT_RATIO, N_DAYS, SEED)
    X, y = report_features(df), df["is_theft"].to_numpy()
    day_cols = [c for c in df.columns if c.startswith("d") and c[1:].isdigit()]

    # ---------- Fig 4.1 class distribution ----------
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = pd.Series(y).value_counts().sort_index()
    bars = ax.bar(["Normal (0)", "Theft (1)"], counts.values, color=[PAL["normal"], PAL["theft"]])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 40, f"{v:,}\n({v / len(y):.0%})", ha="center", fontsize=13)
    ax.set_ylabel("Customers"); ax.set_title("Target Class Distribution"); ax.set_ylim(0, counts.max() * 1.18)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig_4_1_class_distribution.png", dpi=170); plt.close(fig)

    # ---------- Fig 4.2 normal vs bypassed profile ----------
    rng = np.random.default_rng(3)
    nid = rng.choice(np.where(y == 0)[0])
    tid = rng.choice(np.where(df["theft_type"].isin(["intermittent", "night_zero"]))[0])
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, i, title, col in [(axs[0], nid, "Normal consumer", PAL["normal"]),
                              (axs[1], tid, f"Meter bypassed ({df.theft_type[tid]})", PAL["theft"])]:
        s = df.loc[i, day_cols].to_numpy(float)
        ax.plot(range(1, N_DAYS + 1), s, color=col, lw=1.6)
        ax.set_title(title); ax.set_xlabel("Day")
    axs[0].set_ylabel("kWh / day")
    fig.suptitle("Daily Consumption Profiles: Normal vs Meter Bypassed", y=1.03)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig_4_2_consumption_profiles.png", dpi=170, bbox_inches="tight"); plt.close(fig)

    # ---------- Fig 4.3 correlation heatmap ----------
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    corr = pd.concat([X, df["is_theft"]], axis=1).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, cbar_kws={"shrink": .8},
                annot_kws={"size": 10})
    ax.set_title("Correlation Heatmap of Engineered Features")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=10); plt.setp(ax.get_yticklabels(), fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig_4_3_correlation_heatmap.png", dpi=170); plt.close(fig)

    # ---------- split + SMOTE ----------
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    X_sm, y_sm = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "XGBoost Classifier": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08,
                                            subsample=0.9, eval_metric="logloss", random_state=SEED, n_jobs=-1),
    }
    results, probs = {}, {}
    for name, m in models.items():
        m.fit(X_sm, y_sm)
        p = m.predict_proba(X_te)[:, 1]
        pred = (p >= 0.5).astype(int)
        probs[name] = p
        results[name] = {"accuracy": accuracy_score(y_te, pred), "precision": precision_score(y_te, pred),
                         "recall": recall_score(y_te, pred), "f1": f1_score(y_te, pred),
                         "auc_roc": roc_auc_score(y_te, p),
                         "confusion_matrix": confusion_matrix(y_te, pred).tolist()}

    # ---------- unsupervised ----------
    iso = IsolationForest(n_estimators=300, contamination=0.08, random_state=SEED).fit(X_tr)
    iso_pred = (iso.predict(X_te) == -1).astype(int)
    iso_score = -iso.score_samples(X_te)
    top = np.argsort(-iso_score)[: int(0.10 * len(y_te))]
    sc = StandardScaler().fit(X_tr)
    oc = OneClassSVM(nu=0.08, gamma="scale").fit(sc.transform(X_tr))
    oc_pred = (oc.predict(sc.transform(X_te)) == -1).astype(int)
    anomaly = {
        "Isolation Forest": {"precision": precision_score(y_te, iso_pred), "recall": recall_score(y_te, iso_pred),
                             "f1": f1_score(y_te, iso_pred), "precision_at_top10pct": float(y_te[top].mean())},
        "One-Class SVM": {"precision": precision_score(y_te, oc_pred), "recall": recall_score(y_te, oc_pred),
                          "f1": f1_score(y_te, oc_pred)},
    }

    # ---------- Fig 5.1 confusion matrix (XGBoost) ----------
    cm = np.array(results["XGBoost Classifier"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 20},
                xticklabels=["Pred Normal", "Pred Theft"], yticklabels=["Actual Normal", "Actual Theft"])
    ax.set_title(f"Confusion Matrix — XGBoost (n={len(y_te):,})")
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig_5_1_confusion_matrix_xgboost.png", dpi=170); plt.close(fig)

    # ---------- Fig 5.2 ROC ----------
    fig, ax = plt.subplots(figsize=(7, 5.6))
    for (name, p), col in zip(probs.items(), [PAL["blue"], PAL["theft"]]):
        fpr, tpr, _ = roc_curve(y_te, p)
        ax.plot(fpr, tpr, lw=2.6, color=col, label=f"{name} (AUC = {results[name]['auc_roc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1.2)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Classification Models"); ax.legend(loc="lower right", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig_5_2_roc_curves.png", dpi=170); plt.close(fig)

    # ---------- Fig 5.3 feature importance ----------
    imp = pd.Series(models["XGBoost Classifier"].feature_importances_, index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp.index, imp.values * 100, color=PAL["blue"])
    for i, v in enumerate(imp.values * 100):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=12)
    ax.set_xlabel("Relative importance (%)"); ax.set_title("Feature Importance (XGBoost)")
    ax.set_xlim(0, imp.max() * 100 * 1.18)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig_5_3_feature_importance.png", dpi=170); plt.close(fig)

    out = {"dataset": {"customers": N_CUSTOMERS, "days": N_DAYS, "theft_rate": THEFT_RATIO,
                       "test_size": int(len(y_te)), "source": "synthetic (generate_data.py)"},
           "classification": results, "anomaly_detection": anomaly,
           "feature_importance_pct": {k: float(v * 100) for k, v in imp.sort_values(ascending=False).items()},
           "stats_summary": X.describe().round(3).to_dict()}
    with open(f"{FIG_DIR}/report_metrics.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    return out


if __name__ == "__main__":
    o = main()
    print("== Classification (hold-out) ==")
    print(pd.DataFrame(o["classification"]).T.drop(columns="confusion_matrix").round(3).to_string())
    print("\n== Anomaly detection ==")
    print(pd.DataFrame(o["anomaly_detection"]).T.round(3).to_string())
    print("\n== Feature importance (%) ==")
    for k, v in o["feature_importance_pct"].items():
        print(f"  {k:24s} {v:5.1f}")
