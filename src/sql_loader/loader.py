"""
Data Merger + SQL Server Loader
Merges World Bank + WHO data, then loads to SQL Server
"""

import pandas as pd
import pyodbc
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVER   = r"DESKTOP-LR3ML07\MSSQLSERVER01"
DATABASE = "PakistanHealthDB"


def merge_and_clean() -> pd.DataFrame:
    logger.info("Loading raw CSVs...")
    wb  = pd.read_csv("data/raw/world_bank_pakistan.csv")
    who = pd.read_csv("data/raw/who_pakistan.csv")

    df = pd.merge(wb, who[["year","ncd_mortality_30_70_pct","pm25_exposure",
                            "catastrophic_health_exp_10pct","catastrophic_health_exp_25pct"]],
                  on="year", how="left")

    sparse_cols = ["hospital_beds_per_1000","physicians_per_1000",
                   "poverty_headcount_ratio","pm25_exposure",
                   "ncd_mortality_30_70_pct","catastrophic_health_exp_10pct",
                   "catastrophic_health_exp_25pct"]
    for col in sparse_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear", limit_direction="both")

    df["oop_above_who_threshold"] = (df["oop_pct_current_health_exp"] > 40).astype(int)
    df["post_reform_2018"]        = (df["year"] >= 2018).astype(int)
    df["oop_gdp_ratio"]           = (df["oop_pct_current_health_exp"] /
                                     df["current_health_exp_pct_gdp"]).round(4)
    df["disease_burden_index"]    = (
        df["under5_mortality_per_1000"] * 0.4 +
        df["infant_mortality_per_1000"] * 0.3 +
        df["ncd_mortality_30_70_pct"]   * 0.3
    ).round(4)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/merged_pakistan_health.csv", index=False)
    logger.info(f"✅ Merged dataset: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def get_connection():
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={SERVER};"
        f"DATABASE={DATABASE};"
        f"Trusted_Connection=yes;"
    )
    return pyodbc.connect(conn_str)


def create_database():
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={SERVER};"
        f"Trusted_Connection=yes;"
    )
    conn = pyodbc.connect(conn_str, autocommit=True)
    cursor = conn.cursor()
    cursor.execute("""
        IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'PakistanHealthDB')
        CREATE DATABASE PakistanHealthDB
    """)
    cursor.close()
    conn.close()
    logger.info("✅ Database PakistanHealthDB ready")


def create_table(cursor):
    cursor.execute("""
        IF OBJECT_ID('dbo.pakistan_health', 'U') IS NOT NULL
            DROP TABLE dbo.pakistan_health
    """)
    cursor.execute("""
        CREATE TABLE dbo.pakistan_health (
            id                                INT IDENTITY(1,1) PRIMARY KEY,
            country                           NVARCHAR(100),
            country_code                      CHAR(3),
            year                              INT,
            oop_pct_current_health_exp        FLOAT,
            current_health_exp_pct_gdp        FLOAT,
            current_health_exp_per_capita_usd FLOAT,
            under5_mortality_per_1000         FLOAT,
            infant_mortality_per_1000         FLOAT,
            maternal_mortality_per_100k       FLOAT,
            hospital_beds_per_1000            FLOAT,
            physicians_per_1000               FLOAT,
            gdp_per_capita_usd                FLOAT,
            poverty_headcount_ratio           FLOAT,
            ncd_mortality_30_70_pct           FLOAT,
            pm25_exposure                     FLOAT,
            catastrophic_health_exp_10pct     FLOAT,
            catastrophic_health_exp_25pct     FLOAT,
            oop_above_who_threshold           INT,
            post_reform_2018                  INT,
            oop_gdp_ratio                     FLOAT,
            disease_burden_index              FLOAT,
            created_at                        DATETIME DEFAULT GETDATE()
        )
    """)
    logger.info("✅ Table dbo.pakistan_health created")


def load_to_sql(df: pd.DataFrame, conn):
    cursor = conn.cursor()
    create_table(cursor)
    conn.commit()

    cols = [c for c in df.columns if c != "id"]
    placeholders = ", ".join(["?" for _ in cols])
    col_names    = ", ".join([f"[{c}]" for c in cols])
    insert_sql   = f"INSERT INTO dbo.pakistan_health ({col_names}) VALUES ({placeholders})"

    rows_inserted = 0
    for _, row in df[cols].iterrows():
        values = [None if pd.isna(v) else v for v in row]
        cursor.execute(insert_sql, values)
        rows_inserted += 1

    conn.commit()
    logger.info(f"✅ {rows_inserted} rows inserted into dbo.pakistan_health")


if __name__ == "__main__":
    df = merge_and_clean()
    print("\n--- Merged Preview (last 3 rows) ---")
    print(df[["year","oop_pct_current_health_exp","disease_burden_index",
              "oop_above_who_threshold","post_reform_2018"]].tail(3).to_string())

    create_database()
    conn = get_connection()
    load_to_sql(df, conn)
    conn.close()

    print("\n✅ All done! SQL Server Management Studio mein check karein:")
    print("   USE PakistanHealthDB;")
    print("   SELECT * FROM dbo.pakistan_health;")