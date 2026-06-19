"""
Difference-in-Differences (DiD) Causal Inference Model
Research Question: Did Pakistan's 2018 health policy reform causally
reduce OOP health expenditure burden?
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import mlflow
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs("reports", exist_ok=True)
os.makedirs("mlflow_runs", exist_ok=True)

TREATMENT_YEAR = 2018
OUTCOME_1      = "oop_pct_current_health_exp"
OUTCOME_2      = "disease_burden_index"
CONTROL_VARS   = ["gdp_per_capita_usd", "current_health_exp_pct_gdp"]
BLUE           = "#1F4E79"
RED            = "#C00000"
GREEN          = "#70AD47"
GRAY           = "#7F7F7F"


def prepare_did_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["post"]       = (df["year"] >= TREATMENT_YEAR).astype(int)
    df["treated"]    = 1
    df["did"]        = df["post"] * df["treated"]
    df["time_trend"] = df["year"] - df["year"].min()
    return df


def run_did_regression(df: pd.DataFrame, outcome: str):
    df_model = df.dropna(subset=[outcome] + CONTROL_VARS)
    X_cols   = ["post", "did", "time_trend"] + CONTROL_VARS
    X        = sm.add_constant(df_model[X_cols])
    y        = df_model[outcome]
    model    = sm.OLS(y, X).fit(cov_type="HC3")
    return model, df_model


def plot_parallel_trends(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (col, label) in zip(axes, [
        (OUTCOME_1, "OOP % of Health Expenditure"),
        (OUTCOME_2, "Disease Burden Index"),
    ]):
        pre  = df[df["year"] < TREATMENT_YEAR]
        post = df[df["year"] >= TREATMENT_YEAR]

        ax.plot(pre["year"],  pre[col],  "o-", color=RED,   linewidth=2.5,
                markersize=6, label="Pre-Reform (2000–2017)")
        ax.plot(post["year"], post[col], "s-", color=GREEN, linewidth=2.5,
                markersize=6, label="Post-Reform (2018–2023)")
        ax.axvline(TREATMENT_YEAR, color=GRAY, linestyle="--",
                   linewidth=1.8, alpha=0.8, label="Reform 2018")

        pre_slope  = np.polyfit(pre["year"],  pre[col],  1)[0]
        post_slope = np.polyfit(post["year"], post[col], 1)[0]
        ax.text(0.03, 0.95,
                f"Pre-trend:  {pre_slope:+.2f}/yr\nPost-trend: {post_slope:+.2f}/yr",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Year"); ax.set_ylabel(label)
        ax.legend(fontsize=9)

    fig.suptitle("Parallel Trends Test — Pakistan Health Reform 2018",
                 fontsize=13, fontweight="bold", color=BLUE)
    plt.tight_layout()
    plt.savefig("reports/06_parallel_trends.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ reports/06_parallel_trends.png")


def print_did_summary(model, outcome: str):
    print(f"\n{'='*55}")
    print(f"DiD Results — Outcome: {outcome}")
    print(f"{'='*55}")
    print(f"R-squared        : {model.rsquared:.4f}")
    print(f"Adj. R-squared   : {model.rsquared_adj:.4f}")
    print(f"DiD Coefficient  : {model.params['did']:+.4f}")
    print(f"P-value (did)    : {model.pvalues['did']:.4f}  "
          f"{'*** Significant' if model.pvalues['did'] < 0.05 else '(not significant)'}")
    print(f"95% CI           : [{model.conf_int().loc['did', 0]:.3f}, "
          f"{model.conf_int().loc['did', 1]:.3f}]")
    direction = "DECREASE" if model.params["did"] < 0 else "INCREASE"
    print(f"\nInterpretation: 2018 reform caused {abs(model.params['did']):.2f}pp "
          f"{direction} in {outcome}")


def run_with_mlflow(df: pd.DataFrame):
    mlflow.set_tracking_uri("./mlflow_runs")
    mlflow.set_experiment("pakistan_oop_causal_inference")

    df_did = prepare_did_data(df)
    plot_parallel_trends(df_did)

    for outcome in [OUTCOME_1, OUTCOME_2]:
        run_name = f"DiD_{outcome}_reform2018"
        with mlflow.start_run(run_name=run_name):
            model, df_model = run_did_regression(df_did, outcome)

            mlflow.log_param("treatment_year",   TREATMENT_YEAR)
            mlflow.log_param("outcome",          outcome)
            mlflow.log_param("n_obs",            len(df_model))
            mlflow.log_param("controls",         str(CONTROL_VARS))
            mlflow.log_metric("r_squared",       round(model.rsquared, 4))
            mlflow.log_metric("adj_r_squared",   round(model.rsquared_adj, 4))
            mlflow.log_metric("did_coef",        round(model.params["did"], 4))
            mlflow.log_metric("did_pvalue",      round(model.pvalues["did"], 4))
            mlflow.log_metric("did_ci_lower",    round(model.conf_int().loc["did", 0], 4))
            mlflow.log_metric("did_ci_upper",    round(model.conf_int().loc["did", 1], 4))
            mlflow.log_artifact("reports/06_parallel_trends.png")

            print_did_summary(model, outcome)

    return df_did


if __name__ == "__main__":
    df = pd.read_csv("data/processed/merged_pakistan_health.csv")
    print(f"Dataset: {df.shape[0]} rows loaded")
    df_did = run_with_mlflow(df)

    print("\n" + "="*55)
    print("MLflow UI chalane k liye:")
    print("  mlflow ui --backend-store-uri ./mlflow_runs")
    print("  Browser: http://localhost:5000")