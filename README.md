# Pakistan Disease Burden & OOP Health Expenditure Analysis

> *"Do out-of-pocket health payments worsen disease burden in Pakistan? A causal inference approach (2000–2023)"*

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.10-orange)](https://mlflow.org)
[![SQL Server](https://img.shields.io/badge/SQL_Server-2019-red)](https://microsoft.com/sql-server)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-green)](https://streamlit.io)

## Research Question
Pakistan's out-of-pocket (OOP) health spending consistently exceeds the WHO 40% threshold (current: 52.9%). This project investigates whether rising OOP payments causally increase disease burden using **Difference-in-Differences (DiD)** causal inference and machine learning.

## Key Findings
*(Updated after analysis)*

## Project Structure
```
pakistan_oop_disease_burden/
├── data/
│   ├── raw/                   # Downloaded CSVs (not pushed to GitHub)
│   ├── processed/             # Cleaned, merged datasets
│   └── sql/                   # SQL scripts (schema, EDA, analytics)
├── src/
│   ├── scraper/               # World Bank API + WHO GHO scrapers
│   ├── sql_loader/            # SQL Server connection & loaders
│   ├── eda/                   # Exploratory data analysis
│   ├── causal/                # DiD causal inference model
│   ├── models/                # ML training + SHAP explainability
│   └── dashboard/             # Streamlit app
├── notebooks/                 # Jupyter notebooks (step by step)
├── mlflow_runs/               # MLflow experiment tracking
└── reports/                   # Plots, figures for article
```

## Tech Stack
| Layer | Tools |
|---|---|
| Data Collection | World Bank API (`wbgapi`), WHO GHO REST API |
| Database | SQL Server (local), Advanced SQL (CTEs, window functions) |
| EDA | Python, Pandas, Plotly, Seaborn |
| Causal Inference | Difference-in-Differences (DiD), `statsmodels` |
| ML Models | XGBoost, Random Forest, Logistic Regression, SHAP |
| Experiment Tracking | MLflow |
| Dashboard | Streamlit Cloud |

## Setup

### 1. Create conda environment on D: drive
```bash
conda create --prefix D:\envs\pakistan_oop python=3.10
conda activate D:\envs\pakistan_oop
pip install -r requirements.txt
```

### 2. Run data collection
```bash
python src/scraper/world_bank_scraper.py
python src/scraper/who_scraper.py
```

### 3. Load into SQL Server
```bash
python src/sql_loader/loader.py
```
Then run SQL scripts in order: `data/sql/01_create_tables.sql` → `04_advanced_analytics.sql`

### 4. Start MLflow UI
```bash
mlflow ui --backend-store-uri ./mlflow_runs
```
Open: http://localhost:5000

### 5. Run causal inference
```bash
python src/causal/did_model.py
```

### 6. Train ML models
```bash
python src/models/train.py
```

### 7. Launch dashboard
```bash
streamlit run src/dashboard/app.py
```

## Author
**Asif Nawaz** | Healthcare Data Scientist | PMAS Arid Agriculture University Medical Center
- GitHub: [Asif5588-M](https://github.com/Asif5588-M)
- HuggingFace: [asif-nawaz-ml](https://huggingface.co/asif-nawaz-ml)
