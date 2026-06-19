# 🏥 Pakistan Disease Burden & OOP Health Expenditure Analysis

> **"Did Pakistan's 2018 health reform causally reduce out-of-pocket expenditure? A Difference-in-Differences approach (2000–2023)"**

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Live_Dashboard-Streamlit-FF4B4B?logo=streamlit)](https://pakistan-oop-disease-burden-asif-nawaz.streamlit.app)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked-orange?logo=mlflow)](https://mlflow.org)
[![SQL Server](https://img.shields.io/badge/SQL_Server-2019-CC2927?logo=microsoftsqlserver)](https://microsoft.com/sql-server)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🔴 Live Dashboard

**[🚀 pakistan-oop-disease-burden-asif-nawaz.streamlit.app](https://pakistan-oop-disease-burden-asif-nawaz.streamlit.app)**

---

## 📌 Research Question

Pakistan's out-of-pocket (OOP) health spending has consistently exceeded the WHO 40% threshold for 24 consecutive years (2000–2023), averaging **60.8%** — among the highest in South Asia. This project investigates:

1. **Causal question:** Did Pakistan's 2018 Sehat Sahulat Program / NHSRC reform *causally* reduce OOP expenditure?
2. **Disease burden:** Does higher OOP spending worsen disease burden indicators?
3. **Predictive modeling:** Can ML models identify key drivers of OOP spending using SHAP explainability?

---

## 🔑 Key Findings

| Finding | Result |
|---|---|
| Years above WHO 40% threshold | **24 out of 24 (2000–2023)** |
| Average OOP (2000–2023) | **60.8%** of current health expenditure |
| Peak OOP | **72.8%** (2006) |
| Post-2018 Reform DiD Effect | **−6.76 percentage points** |
| DiD P-value | **0.041 (statistically significant)** |
| DiD 95% CI | [−13.24, −0.28] |
| Spearman correlation OOP vs Infant Mortality | **0.378** |
| Best ML Model (XGBoost, 5-Fold CV) | **R² = 0.692, MAE = 2.38pp** |

---

## 🗂️ Project Structure
pakistan_oop_disease_burden/

├── data/

│   ├── processed/                    # Merged cleaned dataset (24 years × 21 vars)

│   └── sql/                          # SQL scripts — schema, EDA, analytics

│       ├── 01_create_tables.sql      # Database schema

│       ├── 03_eda_queries.sql        # Advanced SQL: CTEs, window functions

│       └── 04_advanced_analytics.sql # DiD setup, percentiles, Spearman

├── src/

│   ├── scraper/

│   │   ├── world_bank_scraper.py     # World Bank REST API — 10 indicators

│   │   └── who_scraper.py            # WHO GHO API — 5 indicators

│   ├── sql_loader/

│   │   └── loader.py                 # CSV → SQL Server loader

│   ├── eda/

│   │   └── analysis.py               # 5 publication-quality EDA plots

│   ├── causal/

│   │   └── did_model.py              # DiD regression + MLflow tracking

│   ├── models/

│   │   └── train.py                  # Ridge, RF, XGBoost + SHAP + MLflow

│   └── dashboard/

│       └── app.py                    # Streamlit interactive dashboard

├── reports/                          # Generated plots and figures

├── notebooks/                        # Jupyter notebooks (step-by-step)

├── requirements.txt

└── README.md

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| **Data Collection** | World Bank REST API, WHO GHO API, `requests`, `wbgapi` |
| **Database** | SQL Server 2019, `pyodbc`, Advanced SQL (CTEs, Window Functions, Spearman) |
| **EDA** | Python, Pandas, Plotly, Seaborn, Matplotlib |
| **Causal Inference** | Difference-in-Differences (DiD), `statsmodels`, HC3 robust SE |
| **ML Models** | Ridge Regression, Random Forest, XGBoost, SHAP explainability |
| **Experiment Tracking** | MLflow 2.14 |
| **Dashboard** | Streamlit Cloud |
| **Version Control** | Git, GitHub |

---

## 📊 Methods

### 1. Data Pipeline
- **World Bank REST API:** 10 health & economic indicators for Pakistan (2000–2023)
- **WHO GHO API:** 5 disease burden indicators
- **SQL Server:** Raw + processed schemas with advanced analytical queries
- **Interpolation:** Linear interpolation for sparse indicators (hospital beds, physicians)

### 2. Advanced SQL Analytics
```sql
-- LAG window function: Year-over-year OOP change
SELECT year, oop_pct,
    LAG(oop_pct) OVER (ORDER BY year) AS prev_year,
    oop_pct - LAG(oop_pct) OVER (ORDER BY year) AS yoy_change
FROM pakistan_health;

-- CTE: WHO threshold analysis
WITH threshold_check AS (
    SELECT year,
        CASE WHEN oop_pct > 40 THEN 'ABOVE' ELSE 'BELOW' END AS status
    FROM pakistan_health
)
SELECT status, COUNT(*) AS years, AVG(oop_pct) AS avg_oop
FROM threshold_check GROUP BY status;
```

### 3. Causal Inference — DiD
OOP_it = β0 + β1·Post_t + β2·DiD_t + β3·GDP_it + β4·HealthExp_it + ε_it
Treatment year: 2018 (Sehat Sahulat Program expansion)

DiD coefficient: −6.76pp (p=0.041) ✅

Robust standard errors: HC3

### 4. ML Models (5-Fold CV)
| Model | R² | MAE | RMSE |
|---|---|---|---|
| Ridge Regression | 0.180 | 4.00pp | 4.94pp |
| Random Forest | 0.429 | 2.78pp | 3.41pp |
| **XGBoost** | **0.692** | **2.38pp** | **2.94pp** |

---

## ⚙️ Setup & Reproduction

### 1. Clone & Environment
```bash
git clone https://github.com/Asif5588-M/pakistan-oop-disease-burden.git
cd pakistan-oop-disease-burden

# Create conda environment on D: drive
conda create --prefix D:\envs\pakistan_oop python=3.10
conda activate D:\envs\pakistan_oop
pip install -r requirements.txt
```

### 2. Data Collection
```bash
python src/scraper/world_bank_scraper.py   # World Bank data
python src/scraper/who_scraper.py          # WHO data
python src/sql_loader/loader.py            # Merge + load to SQL Server
```

### 3. Analysis Pipeline
```bash
python src/eda/analysis.py          # EDA plots → reports/
python src/causal/did_model.py      # DiD model + MLflow
python src/models/train.py          # ML models + SHAP
```

### 4. MLflow UI
```bash
mlflow ui --backend-store-uri ./mlflow_runs
# Open: http://localhost:5000
```

### 5. Dashboard
```bash
streamlit run src/dashboard/app.py
# Open: http://localhost:8501
```

---

## 📈 Generated Outputs

| File | Description |
|---|---|
| `reports/01_oop_trend.png` | OOP trend 2000–2023 with WHO threshold |
| `reports/02_oop_vs_disease_burden.png` | OOP vs mortality scatter plots |
| `reports/03_did_visual.png` | DiD pre/post visualization |
| `reports/04_correlation_heatmap.png` | Full correlation matrix |
| `reports/05_spending_vs_oop.png` | Health spending dual-axis |
| `reports/06_parallel_trends.png` | Parallel trends test |
| `reports/07_model_comparison.png` | ML model comparison |
| `reports/08_actual_vs_predicted.png` | Actual vs predicted OOP % |
| `reports/shap_xgboost.png` | SHAP feature importance |

---

## 🔬 Research Implications

1. **Policy:** Despite the 2018 reform causing a significant 6.76pp OOP reduction, Pakistan remains far above the WHO 40% safety threshold — systemic structural reforms are needed.
2. **Health economics:** Persistent high OOP spending correlates with elevated disease burden (Spearman r=0.378 for infant mortality).
3. **Methodology:** DiD causal inference provides more reliable estimates than correlational ML for policy evaluation with limited time-series data (n=24).

---

## 📝 Related Publication

**Nawaz, A., Abdul Rahman, M., Ali, M., & Zafar, N. (2026).** Machine Learning for Sustainable Healthcare: Identifying High-Cost Utilizers and Diagnostic Waste in Pakistan. *Annual Methodological Archive Research Review (AMARR), 4*(3), 740–754. https://doi.org/10.66021/

*Manuscript in preparation for submission to Health Economics Review / BMC Health Services Research.*

---

## 👤 Author

**Asif Nawaz**
Healthcare Data Scientist | PMAS Arid Agriculture University Medical Center, Rawalpindi, Pakistan

[![GitHub](https://img.shields.io/badge/GitHub-Asif5588--M-black?logo=github)](https://github.com/Asif5588-M)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-asif--nawaz--ml-yellow?logo=huggingface)](https://huggingface.co/asif-nawaz-ml)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-asif--nawaz--data--scientist-blue?logo=linkedin)](https://linkedin.com/in/asif-nawaz-data-scientist)

---

## 📄 License

MIT License — feel free to use with attribution.