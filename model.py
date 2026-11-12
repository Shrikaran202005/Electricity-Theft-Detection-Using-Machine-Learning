"""Training, evaluation, persistence and scoring for the theft detector."""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "theft_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


def _candidates():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def _best_threshold(y, p):
    """Threshold maximizing F1 on out-of-fold probabilities."""
    best_t, best_f = 0.5, -1
    for t in np.linspace(0.05, 0.95, 91):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best_f:
            best_t, best_f = t, f
    return float(best_t)


def train(df, save=True):
    X = build_features(df)
    y = df["is_theft"].to_numpy()
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, df.index, test_size=0.25, stratify=y, random_state=42)

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    results, oof = {}, {}
    for name, clf in _candidates().items():
        p = cross_val_predict(clf, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        oof[name] = p
        results[name] = {"cv_roc_auc": float(roc_auc_score(y_tr, p)),
                         "cv_pr_auc": float(average_precision_score(y_tr, p))}
    best_name = max(results, key=lambda k: results[k]["cv_pr_auc"])
    threshold = _best_threshold(y_tr, oof[best_name])

    model = _candidates()[best_name].fit(X_tr, y_tr)
    p_te = model.predict_proba(X_te)[:, 1]
    pred = (p_te >= threshold).astype(int)

    # unsupervised baseline for customers/datasets with no labels
    iso = IsolationForest(n_estimators=200, contamination=0.1, random_state=42).fit(X_tr)

    metrics = {
        "best_model": best_name,
        "threshold": threshold,
        "candidates": results,
        "test": {
            "roc_auc": float(roc_auc_score(y_te, p_te)),
            "pr_auc": float(average_precision_score(y_te, p_te)),
            "precision": float(precision_score(y_te, pred, zero_division=0)),
            "recall": float(recall_score(y_te, pred, zero_division=0)),
            "f1": float(f1_score(y_te, pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
            "n_test": int(len(y_te)),
        },
        "n_customers": int(len(df)),
        "n_features": int(X.shape[1]),
        "theft_rate": float(y.mean()),
    }

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = dict(zip(X.columns, map(float, model.feature_importances_)))
    elif hasattr(model, "named_steps"):
        coefs = model.named_steps["logisticregression"].coef_[0]
        importances = dict(zip(X.columns, map(float, np.abs(coefs))))
    metrics["feature_importance"] = importances

    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump({"model": model, "iso": iso, "threshold": threshold,
                     "features": list(X.columns), "name": best_name,
                     "medians": {"mean_kwh": float(X["mean_kwh"].median()),
                                 "cv": float(X["cv"].median())}}, MODEL_PATH)
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
    return metrics


def load():
    return joblib.load(MODEL_PATH)


def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


REASON_RULES = [
    ("low_day_frac", ">", 0.15, "Many days with abnormally low consumption"),
    ("zero_day_frac", ">", 0.05, "Frequent near-zero readings (possible bypass / meter off)"),
    ("longest_low_run", ">", 25, "Long continuous stretch of very low usage"),
    ("h2_over_h1", "<", 0.75, "Consumption dropped sharply in the second half of the year"),
    ("trend_slope", "<", -0.02, "Persistent downward consumption trend"),
    ("seasonal_r2", "<", 0.10, "Missing normal seasonal pattern"),
    ("mean_vs_peer", "<", 0.55, "Average usage far below peer median"),
    ("cv", ">", 0.55, "Highly erratic consumption"),
]


def explain(feature_row):
    reasons = []
    for f, op, thr, text in REASON_RULES:
        v = feature_row[f]
        if (op == ">" and v > thr) or (op == "<" and v < thr):
            reasons.append(f"{text} ({f} = {v:.2f})")
    return reasons


def score(df, bundle=None):
    """Return df of customer_id, theft_probability, risk, flagged, anomaly_score, reasons."""
    bundle = bundle or load()
    X = build_features(df)[bundle["features"]]
    # re-compute peer features relative to the batch only if it's big enough; else use training medians
    if len(X) < 30:
        X = X.copy()
        X["mean_vs_peer"] = X["mean_kwh"] / bundle["medians"]["mean_kwh"]
        X["cv_vs_peer"] = X["cv"] / bundle["medians"]["cv"]
    p = bundle["model"].predict_proba(X)[:, 1]
    out = pd.DataFrame({
        "customer_id": df["customer_id"].values if "customer_id" in df else np.arange(len(df)),
        "theft_probability": p,
        "flagged": p >= bundle["threshold"],
        "anomaly_score": -bundle["iso"].score_samples(X),
    })
    out["risk"] = pd.cut(out["theft_probability"], [-0.01, 0.3, 0.6, 1.01],
                         labels=["Low", "Medium", "High"])
    out["reasons"] = [" | ".join(explain(X.iloc[i])) or "No clear rule-based indicator"
                      for i in range(len(X))]
    return out


if __name__ == "__main__":
    df = pd.read_csv("data/smart_meter_data.csv")
    m = train(df)
    print(json.dumps({k: v for k, v in m.items() if k != "feature_importance"}, indent=2))
