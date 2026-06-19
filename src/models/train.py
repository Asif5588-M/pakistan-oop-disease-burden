"""
ML Model Training with MLflow Tracking
Models: Logistic Regression, Random Forest, XGBoost
Target: OOP critically high (>55%) vs moderate (40-55%)
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import StratifiedKFold, cross_val_score, LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging, os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs("reports", exist_ok=True)

FEATURES = [
    "current_health_exp_pct_gdp",
    "current_health_exp_per_capita_usd",
    "gdp_per_capita_usd",
    "poverty_headcount_ratio",
    "under5_mortality_per_1000",
    "hospital_beds_per_1000",
    "physicians_per_1000",
    "ncd_mortality_30_70_pct",
    "time_trend",
]
BLUE = "#1F4E79"


def prepare_features(df: pd.DataFrame):
    df = df.copy()
    df["time_trend"] = df["year"] - df["year"].min()

    # New target: critically high OOP (>55%) = 1, moderate (<=55%) = 0
    df["oop_critical"] = (df["oop_pct_current_health_exp"] > 55).astype(int)

    df_clean = df.dropna(subset=FEATURES + ["oop_critical"])
    X = df_clean[FEATURES]
    y = df_clean["oop_critical"]

    logger.info(f"Dataset: {len(X)} samples")
    logger.info(f"Target distribution: Critical OOP>55%: {sum(y==1)} | Moderate: {sum(y==0)}")
    return X, y


def train_and_log(name: str, model, X, y):
    mlflow.set_tracking_uri("./mlflow_runs")
    mlflow.set_experiment("pakistan_oop_ml_models")

    # Small dataset — use LeaveOneOut CV
    loo = LeaveOneOut()

    with mlflow.start_run(run_name=f"{name}_v1"):
        mlflow.log_param("model",      name)
        mlflow.log_param("n_features", len(FEATURES))
        mlflow.log_param("n_samples",  len(X))
        mlflow.log_param("cv_method",  "LeaveOneOut")
        mlflow.log_param("target",     "oop_critical_55pct")

        auc = cross_val_score(model, X, y, cv=loo, scoring="roc_auc")
        f1  = cross_val_score(model, X, y, cv=loo, scoring="f1_macro")
        acc = cross_val_score(model, X, y, cv=loo, scoring="accuracy")

        mlflow.log_metric("loo_auc_mean", round(auc.mean(), 4))
        mlflow.log_metric("loo_f1_mean",  round(f1.mean(),  4))
        mlflow.log_metric("loo_acc_mean", round(acc.mean(), 4))

        model.fit(X, y)
        mlflow.sklearn.log_model(model, name)

        logger.info(f"{name:25s} | AUC: {auc.mean():.4f} "
                    f"| F1: {f1.mean():.4f} | Acc: {acc.mean():.4f}")
    return model, {"auc": round(auc.mean(),4),
                   "f1":  round(f1.mean(),4),
                   "acc": round(acc.mean(),4)}


def shap_plot(model, X: pd.DataFrame, name: str):
    clf = model
    X_plot = X.copy()
    if hasattr(model, "named_steps"):
        clf = model.named_steps["clf"]
        if "scaler" in model.named_steps:
            X_plot = pd.DataFrame(
                model.named_steps["scaler"].transform(X),
                columns=FEATURES
            )
    try:
        explainer = shap.TreeExplainer(clf)
        sv        = explainer.shap_values(X_plot)
        vals      = sv[1] if isinstance(sv, list) else sv

        mean_shap = np.abs(vals).mean(axis=0)
        idx       = np.argsort(mean_shap)

        fig, ax = plt.subplots(figsize=(9, 5))
        colors  = [plt.cm.RdYlGn(v / mean_shap.max()) for v in mean_shap[idx]]
        ax.barh([FEATURES[i] for i in idx], mean_shap[idx], color=colors)
        ax.set_title(f"SHAP Feature Importance — {name}\nPredicting Critical OOP (>55%)",
                     fontweight="bold", color=BLUE)
        ax.set_xlabel("Mean |SHAP value|")
        plt.tight_layout()
        path = f"reports/shap_{name.lower().replace(' ', '_')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"✅ {path}")
    except Exception as e:
        logger.warning(f"SHAP skipped for {name}: {e}")


def model_comparison_plot(results: dict):
    names = list(results.keys())
    aucs  = [results[n]["auc"] for n in names]
    f1s   = [results[n]["f1"]  for n in names]
    accs  = [results[n]["acc"] for n in names]

    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w, aucs, w, label="AUC (LOO)",   color="#1F4E79", alpha=0.85)
    b2 = ax.bar(x,     f1s,  w, label="F1 Macro",    color="#2E75B6", alpha=0.85)
    b3 = ax.bar(x + w, accs, w, label="Accuracy",    color="#9DC3E6", alpha=0.85)

    for bar in [b1, b2, b3]:
        for rect in bar:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.8, color="orange", linestyle="--", alpha=0.6, label="0.8 benchmark")
    ax.set_title("Model Comparison — Pakistan Critical OOP Prediction\n(Leave-One-Out CV, n=24)",
                 fontsize=13, fontweight="bold", color=BLUE)
    ax.set_ylabel("Score"); ax.legend()
    plt.tight_layout()
    plt.savefig("reports/07_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("✅ reports/07_model_comparison.png")


def run_all(csv_path: str = "data/processed/merged_pakistan_health.csv"):
    df = pd.read_csv(csv_path)
    X, y = prepare_features(df)

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=0.5,
                                       class_weight="balanced", random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=3,
            class_weight="balanced", random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.1,
            scale_pos_weight=sum(y==0)/max(sum(y==1), 1),
            eval_metric="auc", random_state=42
        ),
    }

    all_results = {}
    trained     = {}
    for name, model in models.items():
        m, res       = train_and_log(name, model, X, y)
        trained[name] = m
        all_results[name] = res

    print("\n" + "="*55)
    print("MODEL COMPARISON — Critical OOP Prediction (LOO-CV)")
    print("="*55)
    print(f"{'Model':<25} {'AUC':>8} {'F1':>8} {'Acc':>8}")
    print("-"*55)
    for n, r in all_results.items():
        print(f"{n:<25} {r['auc']:>8.4f} {r['f1']:>8.4f} {r['acc']:>8.4f}")

    best = max(all_results, key=lambda x: all_results[x]["auc"])
    print(f"\n🏆 Best: {best} (AUC: {all_results[best]['auc']:.4f})")

    for name in ["Random Forest", "XGBoost"]:
        shap_plot(trained[name], X, name)

    model_comparison_plot(all_results)
    return trained, all_results


if __name__ == "__main__":
    trained, results = run_all()
    print("\n✅ Done! Reports in reports/ | MLflow: http://localhost:5000")