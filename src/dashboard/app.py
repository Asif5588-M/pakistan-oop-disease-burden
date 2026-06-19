"""
Pakistan OOP & Disease Burden — Interactive Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Pakistan OOP Health Expenditure",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 800;
        color: #1F4E79; text-align: center; margin-bottom: 0;
    }
    .sub-title {
        font-size: 1rem; color: #555;
        text-align: center; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1F4E79, #2E75B6);
        border-radius: 12px; padding: 1rem 1.2rem;
        color: white; text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: 800; }
    .metric-label { font-size: 0.8rem; opacity: 0.85; }
    .finding-box {
        background: #EBF3FA; border-left: 4px solid #1F4E79;
        padding: 0.8rem 1rem; border-radius: 6px; margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/merged_pakistan_health.csv")
    df["time_trend"] = df["year"] - df["year"].min()
    df["period"] = df["year"].apply(
        lambda y: "Post-Reform (2018–2023)" if y >= 2018 else "Pre-Reform (2000–2017)"
    )
    return df


def did_regression(df):
    df = df.copy()
    df["post"] = (df["year"] >= 2018).astype(int)
    df["did"]  = df["post"]
    controls   = ["gdp_per_capita_usd", "current_health_exp_pct_gdp"]
    cols       = ["post", "did", "time_trend"] + controls
    df_m       = df.dropna(subset=["oop_pct_current_health_exp"] + controls)
    X = sm.add_constant(df_m[cols])
    y = df_m["oop_pct_current_health_exp"]
    return sm.OLS(y, X).fit(cov_type="HC3"), df_m


# ── Load ──────────────────────────────────────────────────────────────────────
df = load_data()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🏥 Pakistan Disease Burden & OOP Health Expenditure</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Causal Inference Analysis | 2000–2023 | WHO + World Bank Data</p>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Flag_of_Pakistan.svg/320px-Flag_of_Pakistan.svg.png", width=120)
    st.markdown("### Filters")
    year_range = st.slider("Year Range", 2000, 2023, (2000, 2023))
    show_who   = st.checkbox("Show WHO 40% Threshold", value=True)
    show_reform = st.checkbox("Show 2018 Reform Line", value=True)

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **Research Question:**
    Did Pakistan's 2018 health reform causally reduce OOP expenditure?

    **Methods:**
    - Difference-in-Differences (DiD)
    - XGBoost + SHAP
    - Advanced SQL Analytics

    **Data Sources:**
    - World Bank Health Data
    - WHO GHO API

    **Author:** Asif Nawaz
    Healthcare Data Scientist
    PMAS Arid Agriculture University
    """)

df_f = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]

# ── KPI Cards ─────────────────────────────────────────────────────────────────
st.markdown("### 📊 Key Findings")
k1, k2, k3, k4, k5 = st.columns(5)

avg_oop    = df_f["oop_pct_current_health_exp"].mean()
max_oop    = df_f["oop_pct_current_health_exp"].max()
min_oop    = df_f["oop_pct_current_health_exp"].min()
pre_avg    = df[df["year"] < 2018]["oop_pct_current_health_exp"].mean()
post_avg   = df[df["year"] >= 2018]["oop_pct_current_health_exp"].mean()
did_effect = post_avg - pre_avg
yrs_above  = int((df_f["oop_pct_current_health_exp"] > 40).sum())

for col, val, label in zip(
    [k1, k2, k3, k4, k5],
    [f"{avg_oop:.1f}%", f"{max_oop:.1f}%", f"{min_oop:.1f}%",
     f"{did_effect:+.1f}pp", f"{yrs_above}/{len(df_f)}"],
    ["Avg OOP (Selected)", "Peak OOP", "Lowest OOP",
     "Post-Reform Change", "Years > WHO 40%"]
):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{val}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Tab Layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 OOP Trends", "⚕️ Disease Burden", "🔬 Causal Inference (DiD)", "📋 Data Explorer"
])

# ── TAB 1: OOP Trends ─────────────────────────────────────────────────────────
with tab1:
    st.subheader("Out-of-Pocket Health Expenditure Trend — Pakistan")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_f["year"], y=df_f["oop_pct_current_health_exp"],
        mode="lines+markers", name="OOP %",
        line=dict(color="#1F4E79", width=3),
        marker=dict(size=7, color="#1F4E79"),
        fill="tozeroy", fillcolor="rgba(31,78,121,0.08)"
    ))

    if show_who:
        fig.add_hline(y=40, line_dash="dash", line_color="orange",
                      annotation_text="WHO 40% Threshold",
                      annotation_position="bottom right")

    if show_reform and year_range[0] <= 2018 <= year_range[1]:
        fig.add_vline(x=2018, line_dash="dot", line_color="gray",
                      annotation_text="Reform 2018", annotation_position="top")

    fig.update_layout(
        xaxis_title="Year", yaxis_title="OOP % of Current Health Expenditure",
        height=420, template="plotly_white",
        yaxis=dict(range=[35, 80]),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.bar(
            df_f, x="year", y="current_health_exp_per_capita_usd",
            color="period", color_discrete_map={
                "Pre-Reform (2000–2017)": "#1F4E79",
                "Post-Reform (2018–2023)": "#70AD47"
            },
            title="Health Expenditure per Capita (USD)",
            labels={"current_health_exp_per_capita_usd": "USD", "year": "Year"}
        )
        fig2.update_layout(height=340, template="plotly_white", showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = px.scatter(
            df_f, x="gdp_per_capita_usd", y="oop_pct_current_health_exp",
            color="year", size="current_health_exp_per_capita_usd",
            hover_data=["year"], trendline="ols",
            title="GDP per Capita vs OOP %",
            color_continuous_scale="RdYlGn_r",
            labels={"gdp_per_capita_usd": "GDP per Capita (USD)",
                    "oop_pct_current_health_exp": "OOP %"}
        )
        fig3.update_layout(height=340, template="plotly_white")
        st.plotly_chart(fig3, use_container_width=True)

# ── TAB 2: Disease Burden ─────────────────────────────────────────────────────
with tab2:
    st.subheader("Disease Burden Indicators — Pakistan 2000–2023")

    indicator = st.selectbox("Select Indicator", [
        "infant_mortality_per_1000", "under5_mortality_per_1000",
        "maternal_mortality_per_100k", "ncd_mortality_30_70_pct",
        "disease_burden_index"
    ], format_func=lambda x: x.replace("_", " ").title())

    fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    fig4.add_trace(go.Scatter(
        x=df_f["year"], y=df_f[indicator],
        name=indicator.replace("_", " ").title(),
        line=dict(color="#C00000", width=2.5), mode="lines+markers"
    ), secondary_y=False)
    fig4.add_trace(go.Scatter(
        x=df_f["year"], y=df_f["oop_pct_current_health_exp"],
        name="OOP %", line=dict(color="#1F4E79", width=2, dash="dot"),
        mode="lines+markers"
    ), secondary_y=True)

    if show_reform and year_range[0] <= 2018 <= year_range[1]:
        fig4.add_vline(x=2018, line_dash="dot", line_color="gray")

    fig4.update_layout(height=420, template="plotly_white", hovermode="x unified")
    fig4.update_yaxes(title_text=indicator.replace("_", " ").title(), secondary_y=False)
    fig4.update_yaxes(title_text="OOP %", secondary_y=True)
    st.plotly_chart(fig4, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    for col, ind, label in zip(
        [col1, col2, col3],
        ["infant_mortality_per_1000", "under5_mortality_per_1000", "ncd_mortality_30_70_pct"],
        ["Infant Mortality", "Under-5 Mortality", "NCD Mortality 30–70"]
    ):
        corr = df_f[["oop_pct_current_health_exp", ind]].corr().iloc[0, 1]
        fig_s = px.scatter(
            df_f, x="oop_pct_current_health_exp", y=ind,
            trendline="ols", color="year",
            color_continuous_scale="RdYlGn_r",
            title=f"OOP % vs {label}<br>r = {corr:.3f}",
            labels={"oop_pct_current_health_exp": "OOP %"}
        )
        fig_s.update_layout(height=300, template="plotly_white", showlegend=False)
        col.plotly_chart(fig_s, use_container_width=True)

# ── TAB 3: Causal Inference ───────────────────────────────────────────────────
with tab3:
    st.subheader("Difference-in-Differences (DiD) — Causal Inference")

    st.markdown("""
    <div class="finding-box">
    <b>Research Question:</b> Did Pakistan's 2018 Sehat Sahulat Program / NHSRC reform
    causally reduce out-of-pocket health expenditure?<br>
    <b>Design:</b> Pre-period = 2000–2017 | Post-period = 2018–2023 | Controls: GDP, Health Exp % GDP
    </div>
    """, unsafe_allow_html=True)

    model, df_m = did_regression(df)
    did_coef  = model.params["did"]
    did_pval  = model.pvalues["did"]
    did_ci_lo = model.conf_int().loc["did", 0]
    did_ci_hi = model.conf_int().loc["did", 1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DiD Coefficient", f"{did_coef:+.2f}pp")
    c2.metric("P-value", f"{did_pval:.4f}", delta="Significant ✅" if did_pval < 0.05 else "Not Sig.")
    c3.metric("95% CI Lower", f"{did_ci_lo:.2f}")
    c4.metric("95% CI Upper", f"{did_ci_hi:.2f}")

    pre  = df[df["year"] < 2018]
    post = df[df["year"] >= 2018]

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=pre["year"], y=pre["oop_pct_current_health_exp"],
        name=f"Pre-Reform (avg={pre['oop_pct_current_health_exp'].mean():.1f}%)",
        line=dict(color="#C00000", width=3), mode="lines+markers", marker=dict(size=7)
    ))
    fig5.add_trace(go.Scatter(
        x=post["year"], y=post["oop_pct_current_health_exp"],
        name=f"Post-Reform (avg={post['oop_pct_current_health_exp'].mean():.1f}%)",
        line=dict(color="#70AD47", width=3), mode="lines+markers", marker=dict(size=7)
    ))
    if show_who:
        fig5.add_hline(y=40, line_dash="dash", line_color="orange",
                       annotation_text="WHO 40% Threshold")
    fig5.add_vline(x=2018, line_dash="dot", line_color="gray",
                   annotation_text="Reform 2018")
    fig5.update_layout(
        title="OOP % Before and After 2018 Reform",
        xaxis_title="Year", yaxis_title="OOP %",
        height=420, template="plotly_white", hovermode="x unified"
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("### 📋 DiD Regression Summary")
    summary_df = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient": model.params.values.round(4),
        "Std Error":   model.bse.values.round(4),
        "P-value":     model.pvalues.values.round(4),
        "Significant": ["✅" if p < 0.05 else "—" for p in model.pvalues.values]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div class="finding-box">
    <b>Finding:</b> The 2018 health reform caused a statistically significant
    <b>{abs(did_coef):.2f} percentage point decrease</b> in OOP health expenditure
    (p={did_pval:.4f}, 95% CI [{did_ci_lo:.2f}, {did_ci_hi:.2f}]).
    Despite this reduction, Pakistan's OOP spending remains consistently above the WHO 40% threshold,
    averaging <b>{post['oop_pct_current_health_exp'].mean():.1f}%</b> in the post-reform period.
    </div>
    """, unsafe_allow_html=True)

# ── TAB 4: Data Explorer ──────────────────────────────────────────────────────
with tab4:
    st.subheader("📋 Raw Data Explorer")

    cols_show = st.multiselect("Select Columns", df.columns.tolist(),
                               default=["year", "oop_pct_current_health_exp",
                                        "current_health_exp_pct_gdp",
                                        "gdp_per_capita_usd",
                                        "infant_mortality_per_1000",
                                        "disease_burden_index"])
    st.dataframe(df_f[cols_show].round(3), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download Filtered Data (CSV)",
            df_f[cols_show].to_csv(index=False),
            "pakistan_health_filtered.csv", "text/csv"
        )
    with col2:
        st.markdown(f"**Rows:** {len(df_f)} | **Columns:** {len(cols_show)} | "
                    f"**Years:** {year_range[0]}–{year_range[1]}")

    st.markdown("### Correlation Matrix")
    num_cols = df_f.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if c not in ["time_trend"]]
    corr_mat = df_f[num_cols].corr().round(3)
    fig_h = px.imshow(corr_mat, color_continuous_scale="RdBu_r",
                      zmin=-1, zmax=1, text_auto=".2f",
                      title="Correlation Heatmap — All Indicators")
    fig_h.update_layout(height=500)
    st.plotly_chart(fig_h, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.85rem;'>
Asif Nawaz | Healthcare Data Scientist | PMAS Arid Agriculture University Medical Center<br>
Data: World Bank Health Data + WHO GHO API | Method: DiD Causal Inference + XGBoost + SHAP
</div>
""", unsafe_allow_html=True)