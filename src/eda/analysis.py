"""
EDA + Visualization
Pakistan OOP & Disease Burden Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

os.makedirs("reports", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")
BLUE   = "#1F4E79"
RED    = "#C00000"
ORANGE = "#ED7D31"
GREEN  = "#70AD47"
GRAY   = "#7F7F7F"


def load_data() -> pd.DataFrame:
    return pd.read_csv("data/processed/merged_pakistan_health.csv")


# ── Plot 1: OOP Trend with WHO threshold ──────────────────────────────────────
def plot_oop_trend(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.fill_between(df["year"], df["oop_pct_current_health_exp"], 40,
                    where=df["oop_pct_current_health_exp"] > 40,
                    alpha=0.15, color=RED, label="Above WHO threshold")
    ax.plot(df["year"], df["oop_pct_current_health_exp"],
            color=BLUE, linewidth=2.5, marker="o", markersize=5, label="OOP % of Health Exp")
    ax.axhline(40, color=ORANGE, linewidth=1.8, linestyle="--", label="WHO 40% Threshold")
    ax.axvline(2018, color=GRAY, linewidth=1.5, linestyle=":", alpha=0.8)
    ax.text(2018.2, ax.get_ylim()[0] + 2, "Reform\n2018", color=GRAY, fontsize=9)

    ax.set_title("Pakistan Out-of-Pocket Health Expenditure (2000–2023)\nExceeds WHO 40% Threshold Every Year",
                 fontsize=13, fontweight="bold", color=BLUE)
    ax.set_xlabel("Year"); ax.set_ylabel("OOP % of Current Health Expenditure")
    ax.legend(); ax.set_xlim(2000, 2023)
    plt.tight_layout()
    plt.savefig("reports/01_oop_trend.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ reports/01_oop_trend.png")


# ── Plot 2: Disease Burden vs OOP scatter ─────────────────────────────────────
def plot_oop_vs_disease_burden(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    pairs = [
        ("oop_pct_current_health_exp", "infant_mortality_per_1000",
         "Infant Mortality (per 1,000)"),
        ("oop_pct_current_health_exp", "under5_mortality_per_1000",
         "Under-5 Mortality (per 1,000)"),
        ("oop_pct_current_health_exp", "ncd_mortality_30_70_pct",
         "NCD Mortality 30-70 (%)"),
    ]

    for ax, (x, y, ylabel) in zip(axes, pairs):
        sc = ax.scatter(df[x], df[y], c=df["year"], cmap="RdYlGn_r",
                        s=80, edgecolors="white", linewidths=0.5)
        # Trend line
        z = np.polyfit(df[x].dropna(), df[y].dropna(), 1)
        p = np.poly1d(z)
        xr = np.linspace(df[x].min(), df[x].max(), 100)
        ax.plot(xr, p(xr), color=BLUE, linewidth=1.5, linestyle="--", alpha=0.7)
        corr = df[[x, y]].corr().iloc[0, 1]
        ax.set_title(f"r = {corr:.3f}", fontsize=11, color=BLUE)
        ax.set_xlabel("OOP %"); ax.set_ylabel(ylabel)
        plt.colorbar(sc, ax=ax, label="Year")

    fig.suptitle("OOP Health Expenditure vs Disease Burden Indicators — Pakistan 2000–2023",
                 fontsize=13, fontweight="bold", color=BLUE, y=1.02)
    plt.tight_layout()
    plt.savefig("reports/02_oop_vs_disease_burden.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ reports/02_oop_vs_disease_burden.png")


# ── Plot 3: DiD Visual — Pre vs Post Reform ───────────────────────────────────
def plot_did_visual(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (col, label) in zip(axes, [
        ("oop_pct_current_health_exp", "OOP % of Health Expenditure"),
        ("disease_burden_index",       "Disease Burden Index"),
    ]):
        pre  = df[df["year"] < 2018]
        post = df[df["year"] >= 2018]
        ax.plot(pre["year"],  pre[col],  "o-", color=RED,   linewidth=2.5,
                label=f"Pre-Reform  (avg={pre[col].mean():.1f})")
        ax.plot(post["year"], post[col], "s-", color=GREEN, linewidth=2.5,
                label=f"Post-Reform (avg={post[col].mean():.1f})")
        ax.axvline(2018, color=GRAY, linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(2018.1, df[col].max() * 0.97, "Policy\nReform", color=GRAY, fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Year"); ax.set_ylabel(label)
        ax.legend(fontsize=9)

    fig.suptitle("Difference-in-Differences: Impact of 2018 Health Reform — Pakistan",
                 fontsize=13, fontweight="bold", color=BLUE)
    plt.tight_layout()
    plt.savefig("reports/03_did_visual.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ reports/03_did_visual.png")


# ── Plot 4: Correlation Heatmap ───────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame):
    cols = ["oop_pct_current_health_exp", "current_health_exp_pct_gdp",
            "gdp_per_capita_usd", "infant_mortality_per_1000",
            "under5_mortality_per_1000", "maternal_mortality_per_100k",
            "ncd_mortality_30_70_pct", "disease_burden_index",
            "hospital_beds_per_1000", "physicians_per_1000"]

    corr = df[cols].corr()
    labels = ["OOP%", "Health Exp%GDP", "GDP/capita",
              "Infant Mort", "U5 Mort", "Maternal Mort",
              "NCD Mort", "Disease Index", "Beds/1000", "Physicians"]

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, linewidths=0.5,
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 9})
    ax.set_title("Correlation Matrix — Pakistan Health Indicators (2000–2023)",
                 fontsize=13, fontweight="bold", color=BLUE, pad=15)
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("reports/04_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ reports/04_correlation_heatmap.png")


# ── Plot 5: OOP Components Over Time ──────────────────────────────────────────
def plot_health_spending_trend(df: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax2 = ax1.twinx()

    ax1.bar(df["year"], df["current_health_exp_per_capita_usd"],
            color=BLUE, alpha=0.6, label="Health Exp per Capita (USD)")
    ax2.plot(df["year"], df["oop_pct_current_health_exp"],
             color=RED, linewidth=2.5, marker="o", markersize=5,
             label="OOP % (right axis)")
    ax2.axhline(40, color=ORANGE, linestyle="--", linewidth=1.5, alpha=0.8)

    ax1.set_xlabel("Year")
    ax1.set_ylabel("Health Expenditure per Capita (USD)", color=BLUE)
    ax2.set_ylabel("OOP % of Health Expenditure", color=RED)
    ax1.set_title("Pakistan: Health Spending vs OOP Burden (2000–2023)",
                  fontsize=13, fontweight="bold", color=BLUE)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    plt.tight_layout()
    plt.savefig("reports/05_spending_vs_oop.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ reports/05_spending_vs_oop.png")


if __name__ == "__main__":
    df = load_data()
    print(f"Data loaded: {df.shape}")
    print(f"Years: {df['year'].min()}–{df['year'].max()}")
    print(f"OOP range: {df['oop_pct_current_health_exp'].min():.1f}% – {df['oop_pct_current_health_exp'].max():.1f}%")
    print("\nGenerating plots...")
    plot_oop_trend(df)
    plot_oop_vs_disease_burden(df)
    plot_did_visual(df)
    plot_correlation_heatmap(df)
    plot_health_spending_trend(df)
    print("\n✅ All 5 plots saved in reports/ folder")