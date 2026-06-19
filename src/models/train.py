"""
ML Model Training with MLflow Tracking
Models: Logistic Regression, Random Forest, XGBoost
Target: Binary — Catastrophic Health Expenditure (OOP > 40%)
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import logging, os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURES = [
    "health_exp_pct_gdp", "health_exp_per_capita", "gdp_per_capita",
    "poverty_ratio", "under5_mortality", "hospital_beds", "physicians",
    "ncd_mortality", "time_trend"
]
TARGET = "oop_above_who_threshold"


def prepare_features(df: pd.DataFrame):
    df = df.copy()
    df["time_trend"] = df["year"] - df["year"].min()
    df[TARGET] = (df["oop_pct_health_exp"] > 40).astype(int)
    df_clean = df.dropna(subset=FEATURES + [TARGET])
    X = df_clean[FEATURES]
    y = df_clean[TARGET]
    return X, y, df_clean


def train_model(model_name: str, model, X, y, run_name: str):
    mlflow.set_experiment("pakistan_oop_ml_models")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model", model_name)
        mlflow.log_param("n_features", len(FEATURES))
        mlflow.log_param("n_samples", len(X))

        # Cross-validated metrics
        auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        f1_scores  = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")

        mlflow.log_metric("cv_auc_mean",  round(auc_scores.mean(), 4))
        mlflow.log_metric("cv_auc_std",   round(auc_scores.std(), 4))
        mlflow.log_metric("cv_f1_mean",   round(f1_scores.mean(), 4))

        # Fit on full data for SHAP
        model.fit(X, y)
        mlflow.sklearn.log_model(model, model_name)

        logger.info(f"{model_name} | AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f} | F1: {f1_scores.mean():.4f}")
    return model


def shap_analysis(model, X: pd.DataFrame, model_name: str):
    os.makedirs("reports", exist_ok=True)
    if hasattr(model, "named_steps"):  # Pipeline
        clf = model.named_steps.get("clf", model)
        X_transformed = model.named_steps["scaler"].transform(X) if "scaler" in model.named_steps else X
        X_plot = pd.DataFrame(X_transformed, columns=FEATURES)
    else:
        clf = model
        X_plot = X

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_plot)
    vals = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(vals, X_plot, feature_names=FEATURES, show=False, plot_type="bar")
    plt.title(f"SHAP Feature Importance — {model_name}", fontweight="bold")
    plt.tight_layout()
    path = f"reports/shap_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP plot saved → {path}")
    return path


def run_all_models(csv_path: str = "data/processed/merged_pakistan_health.csv"):
    df = pd.read_csv(csv_path)
    X, y, df_clean = prepare_features(df)
    logger.info(f"Dataset: {len(X)} samples | Target distribution:\n{y.value_counts()}")

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced"))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=5, class_weight="balanced", random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            scale_pos_weight=sum(y==0)/sum(y==1), eval_metric="auc",
            use_label_encoder=False, random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        trained = train_model(name, model, X, y, run_name=f"{name}_v1")
        results[name] = trained

    # SHAP for tree models
    shap_analysis(results["XGBoost"], X, "XGBoost")
    return results


if __name__ == "__main__":
    run_all_models()
