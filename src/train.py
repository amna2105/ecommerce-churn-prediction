"""
train.py
--------
End-to-end training script: loads data, builds features, trains and compares
several models for churn prediction, and saves the best one to models/.

Run from the project root:
    python -m src.train
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data_preprocessing import build_features, load_and_clean_data

DATA_PATH = "data/raw/E_Commerce_Dataset.xlsx"
MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")
RANDOM_STATE = 42


def get_models(X_train_balanced, y_train_balanced):
    return {
        "Logistic Regression (SMOTE)": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000
        ).fit(X_train_balanced, y_train_balanced),
        "Random Forest (SMOTE)": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ).fit(X_train_balanced, y_train_balanced),
    }


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", cm)
    print(f"AUC-ROC: {auc:.4f}")

    return {
        "name": name,
        "model": model,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "auc": auc,
        "f1": f1,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "confusion_matrix": cm.tolist(),
    }


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and cleaning data...")
    df = load_and_clean_data(DATA_PATH)
    churn_rate = round(df["Churn"].mean() * 100, 2)
    print(f"Overall churn rate: {churn_rate}%")

    X, y, feature_names, scaler = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("Balancing training set with SMOTE...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    models = get_models(X_train_bal, y_train_bal)

    results = [evaluate(name, model, X_test, y_test) for name, model in models.items()]
    best = max(results, key=lambda r: r["auc"])
    print(f"\nBest model by AUC-ROC: {best['name']} (AUC={best['auc']:.4f})")

    # --- Save artifacts -----------------------------------------------------
    joblib.dump(best["model"], MODELS_DIR / "churn_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")

    metrics_summary = {
        "churn_rate_pct": churn_rate,
        "n_samples": int(len(df)),
        "n_features": len(feature_names),
        "results": [
            {k: v for k, v in r.items() if k not in ("model", "y_pred", "y_prob")}
            for r in results
        ],
        "best_model": best["name"],
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # --- Plots ---------------------------------------------------------------
    plt.figure(figsize=(6, 5))
    for r in results:
        RocCurveDisplay.from_predictions(y_test, r["y_prob"], name=r["name"], ax=plt.gca())
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve Comparison")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve_comparison.png", dpi=150)
    plt.close()

    cm = np.array(best["confusion_matrix"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix — {best['name']}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    if hasattr(best["model"], "feature_importances_"):
        importances = pd.Series(best["model"].feature_importances_, index=feature_names)
        top15 = importances.sort_values(ascending=False).head(15)
        plt.figure(figsize=(8, 6))
        sns.barplot(x=top15.values, y=top15.index, color="steelblue")
        plt.title("Top 15 Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
        plt.close()

    print(f"\nSaved model, scaler, metrics, and figures to {MODELS_DIR}/ and {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
