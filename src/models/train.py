"""
ML Model Training — Regression Approach
Target: Predict OOP % of health expenditure (continuous)
Models: Ridge, Random Forest, XGBoost + SHAP explainability
MLflow tracked
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, logging, os
warnings.filterwarnings("ignore")

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
TARGET = "oop_pct_current_health_exp"
BLUE   = "#1F4E79"


def prepare_data(df: pd.DataFrame):
    df = df.copy()
    df["time_trend"] = df["year"] - df["year"].min()
    df_clean = df.dropna(subset=FEATURES + [TARGET])
    X = df_clean[FEATURES]
    y = df_clean[TARGET]
    logger.info(f"Dataset: {len(X)} samples | Target range: {y.min():.1f}% – {y.max():.1f}%")
    return X, y


def train_and_log(name: str, model, X, y):
    mlflow.set_tracking_uri("./mlflow_runs")
    mlflow.set_experiment("pakistan_oop_ml_models")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    with mlflow.start_run(run_name=f"{name}_regression_v1"):
        mlflow.log_param("model",      name)
        mlflow.log_param("task",       "regression")
        mlflow.log_param("target",     TARGET)
        mlflow.log_param("n_features", len(FEATURES))
        mlflow.log_param("n_samples",  len(X))

        cv_results = cross_validate(
            model, X, y, cv=cv,
            scoring={"r2": "r2", "mae": "neg_mean_absolute_error",
                     "rmse": "neg_root_mean_squared_error"},
            return_train_score=False
        )

        r2   = cv_results["test_r2"].mean()
        mae  = -cv_results["test_mae"].mean()
        rmse = -cv_results["test_rmse"].mean()

        mlflow.log_metric("cv_r2_mean",   round(r2,   4))
        mlflow.log_metric("cv_mae_mean",  round(mae,  4))
        mlflow.log_metric("cv_rmse_mean", round(rmse, 4))

        model.fit(X, y)
        mlflow.sklearn.log_model(model, name)

        logger.info(f"{name:25s} | R²: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    return model, {"r2": round(r2,4), "mae": round(mae,4), "rmse": round(rmse,4)}


def shap_plot(model, X: pd.DataFrame, name: str):
    clf   = model
    X_plt = X.copy()

    if hasattr(model, "named_steps"):
        clf = model.named_steps["clf"]
        if "scaler" in model.named_steps:
            X_plt = pd.DataFrame(
                model.named_steps["scaler"].transform(X),
                columns=FEATURES
            )

    try:
        explainer = shap.TreeExplainer(clf)
        sv        = explainer.shap_values(X_plt)

        mean_shap = np.abs(sv).mean(axis=0)
        idx       = np.argsort(mean_shap)
        feat_labels = [f.replace("_", " ").title() for f in FEATURES]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart
        colors = plt.cm.RdYlGn_r(mean_shap[idx] / mean_shap.max())
        axes[0].barh([feat_labels[i] for i in idx], mean_shap[idx], color=colors)
        axes[0].set_title(f"SHAP Feature Importance\n{name} — Predicting OOP %",
                          fontweight="bold", color=BLUE)
        axes[0].set_xlabel("Mean |SHAP value| (percentage points)")

        # Scatter: actual vs predicted
        y_pred = clf.predict(X_plt)
        axes[1].scatter(X["time_trend"] + 2000, y_pred,
                        color=BLUE, s=60, label="Predicted", alpha=0.8)
        axes[1].plot(X["time_trend"] + 2000,
                     [TARGET] * len(X) if isinstance(TARGET, float) else [0]*len(X),
                     color="gray", linewidth=0)
        axes[1].set_title(f"Predicted OOP % Over Time\n{name}",
                          fontweight="bold", color=BLUE)
        axes[1].set_xlabel("Year"); axes[1].set_ylabel("Predicted OOP %")

        plt.tight_layout()
        path = f"reports/shap_{name.lower().replace(' ', '_')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"✅ {path}")
    except Exception as e:
        logger.warning(f"SHAP skipped for {name}: {e}")


def actual_vs_predicted_plot(models_dict: dict, X, y):
    fig, axes = plt.subplots(1, len(models_dict), figsize=(15, 4))
    years = X["time_trend"].values + 2000

    for ax, (name, model) in zip(axes, models_dict.items()):
        clf   = model
        X_plt = X.copy()
        if hasattr(model, "named_steps"):
            clf = model.named_steps["clf"]
            if "scaler" in model.named_steps:
                X_plt = pd.DataFrame(
                    model.named_steps["scaler"].transform(X), columns=FEATURES)

        y_pred = model.predict(X)
        r2     = r2_score(y, y_pred)
        mae    = mean_absolute_error(y, y_pred)

        ax.plot(years, y.values,   "o-", color="#C00000", lw=2, label="Actual",    ms=5)
        ax.plot(years, y_pred,     "s--",color=BLUE,      lw=2, label="Predicted", ms=5)
        ax.axhline(40, color="orange", linestyle=":", alpha=0.7, label="WHO 40%")
        ax.set_title(f"{name}\nR²={r2:.3f} | MAE={mae:.2f}pp",
                     fontweight="bold", fontsize=10)
        ax.set_xlabel("Year"); ax.set_ylabel("OOP %")
        ax.legend(fontsize=8)

    fig.suptitle("Actual vs Predicted OOP % — Pakistan 2000–2023",
                 fontsize=13, fontweight="bold", color=BLUE)
    plt.tight_layout()
    plt.savefig("reports/08_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("✅ reports/08_actual_vs_predicted.png")


def model_comparison_plot(results: dict):
    names = list(results.keys())
    r2s   = [results[n]["r2"]   for n in names]
    maes  = [results[n]["mae"]  for n in names]
    rmses = [results[n]["rmse"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = ["#1F4E79", "#2E75B6", "#9DC3E6"]

    for ax, vals, label, fmt in zip(
        axes,
        [r2s, maes, rmses],
        ["R² Score (higher=better)", "MAE — pp (lower=better)", "RMSE — pp (lower=better)"],
        [".3f", ".2f", ".2f"]
    ):
        bars = ax.bar(names, vals, color=colors, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:{fmt}}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title(label, fontweight="bold", color=BLUE)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Model Comparison — OOP % Regression (5-Fold CV)\nPakistan 2000–2023",
                 fontsize=13, fontweight="bold", color=BLUE)
    plt.tight_layout()
    plt.savefig("reports/07_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("✅ reports/07_model_comparison.png")


def run_all(csv_path: str = "data/processed/merged_pakistan_health.csv"):
    df = pd.read_csv(csv_path)
    X, y = prepare_data(df)

    models = {
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", Ridge(alpha=1.0))
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=4, random_state=42
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="rmse", random_state=42
        ),
    }

    all_results = {}
    trained     = {}

    for name, model in models.items():
        m, res          = train_and_log(name, model, X, y)
        trained[name]   = m
        all_results[name] = res

    print("\n" + "="*58)
    print("MODEL COMPARISON — OOP % Regression (5-Fold CV)")
    print("="*58)
    print(f"{'Model':<25} {'R²':>8} {'MAE(pp)':>10} {'RMSE(pp)':>10}")
    print("-"*58)
    for n, r in all_results.items():
        print(f"{n:<25} {r['r2']:>8.4f} {r['mae']:>10.4f} {r['rmse']:>10.4f}")

    best = max(all_results, key=lambda x: all_results[x]["r2"])
    print(f"\n🏆 Best Model: {best} (R²: {all_results[best]['r2']:.4f})")

    for name in ["Random Forest", "XGBoost"]:
        shap_plot(trained[name], X, name)

    actual_vs_predicted_plot(trained, X, y)
    model_comparison_plot(all_results)

    return trained, all_results


if __name__ == "__main__":
    trained, results = run_all()
    print("\n✅ All done!")
    print("✅ Plots in reports/")
    print("✅ MLflow: http://localhost:5000")