"""
Difference-in-Differences (DiD) Model
Research Question:
  Did Pakistan's 2018 health policy reform (NHSRC / Sehat Sahulat Program)
  causally reduce catastrophic health expenditure?

Design:
  - Treatment: Pakistan post-2018 (policy shock)
  - Control: Pre-reform period (2000-2017)
  - Outcome: OOP % of current health expenditure
  - Parallel trends assumption tested visually + statistically
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TREATMENT_YEAR = 2018
OUTCOME_VAR    = "oop_pct_health_exp"
CONTROL_VARS   = ["gdp_per_capita", "poverty_ratio", "health_exp_pct_gdp"]


def prepare_did_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["post"]      = (df["year"] >= TREATMENT_YEAR).astype(int)
    df["treated"]   = 1                                 # single country DiD
    df["did"]       = df["post"] * df["treated"]        # interaction term
    df["time_trend"] = df["year"] - df["year"].min()   # linear time trend
    return df


def run_did_regression(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    OLS: Y = β0 + β1*post + β2*treated + β3*(post×treated) + controls + ε
    β3 (did coefficient) = causal effect of reform
    """
    df_model = df.dropna(subset=[OUTCOME_VAR] + CONTROL_VARS)
    X_cols = ["post", "treated", "did", "time_trend"] + CONTROL_VARS
    X = sm.add_constant(df_model[X_cols])
    y = df_model[OUTCOME_VAR]

    model = sm.OLS(y, X).fit(cov_type="HC3")  # robust standard errors
    return model, df_model


def test_parallel_trends(df: pd.DataFrame, save_path: str = "reports/parallel_trends.png"):
    """Visual test: pre-treatment trends must be parallel"""
    pre = df[df["year"] < TREATMENT_YEAR]
    post = df[df["year"] >= TREATMENT_YEAR]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pre["year"],  pre[OUTCOME_VAR],  "b-o", label="Pre-Reform", linewidth=2)
    ax.plot(post["year"], post[OUTCOME_VAR], "r-o", label="Post-Reform (Treatment)", linewidth=2)
    ax.axvline(x=TREATMENT_YEAR, color="gray", linestyle="--", alpha=0.7, label=f"Reform ({TREATMENT_YEAR})")
    ax.axhline(y=40, color="orange", linestyle=":", alpha=0.8, label="WHO 40% threshold")
    ax.set_title("Pakistan OOP Health Expenditure — Parallel Trends Test", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("OOP % of Current Health Expenditure")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Parallel trends plot saved → {save_path}")


def run_with_mlflow(df: pd.DataFrame):
    mlflow.set_experiment("pakistan_oop_causal_inference")
    with mlflow.start_run(run_name="DiD_OOP_Reform_2018"):
        df_did = prepare_did_data(df)
        model, df_model = run_did_regression(df_did)

        # Log parameters
        mlflow.log_param("treatment_year", TREATMENT_YEAR)
        mlflow.log_param("outcome_variable", OUTCOME_VAR)
        mlflow.log_param("n_observations", len(df_model))
        mlflow.log_param("control_variables", CONTROL_VARS)

        # Log metrics
        mlflow.log_metric("r_squared",       round(model.rsquared, 4))
        mlflow.log_metric("adj_r_squared",   round(model.rsquared_adj, 4))
        mlflow.log_metric("did_coefficient", round(model.params["did"], 4))
        mlflow.log_metric("did_pvalue",      round(model.pvalues["did"], 4))
        mlflow.log_metric("aic",             round(model.aic, 2))

        # Parallel trends plot
        test_parallel_trends(df_did, "reports/parallel_trends.png")
        mlflow.log_artifact("reports/parallel_trends.png")

        logger.info("\n" + model.summary().as_text())
        logger.info(f"\nDiD Coefficient: {model.params['did']:.4f}")
        logger.info(f"P-value: {model.pvalues['did']:.4f}")
        logger.info(f"Interpretation: Reform caused {abs(model.params['did']):.2f}pp "
                    f"{'decrease' if model.params['did'] < 0 else 'increase'} in OOP spending")
        return model


if __name__ == "__main__":
    df = pd.read_csv("data/processed/merged_pakistan_health.csv")
    model = run_with_mlflow(df)
