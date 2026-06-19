"""
World Bank Data Scraper — Fixed Version
Uses direct REST API instead of wbgapi library
"""

import requests
import pandas as pd
import os
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INDICATORS = {
    "SH.XPD.OOPC.CH.ZS": "oop_pct_current_health_exp",
    "SH.XPD.CHEX.GD.ZS":  "current_health_exp_pct_gdp",
    "SH.XPD.CHEX.PC.CD":  "current_health_exp_per_capita_usd",
    "SH.DYN.MORT":         "under5_mortality_per_1000",
    "SP.DYN.IMRT.IN":      "infant_mortality_per_1000",
    "SH.STA.MMRT":         "maternal_mortality_per_100k",
    "SH.MED.BEDS.ZS":      "hospital_beds_per_1000",
    "SH.MED.PHYS.ZS":      "physicians_per_1000",
    "NY.GDP.PCAP.CD":      "gdp_per_capita_usd",
    "SI.POV.NAHC":         "poverty_headcount_ratio",
}

COUNTRY = "PK"
START_YEAR = 2000
END_YEAR = 2023


def fetch_indicator(code: str, col_name: str) -> pd.DataFrame:
    """Fetch single indicator using World Bank REST API v2"""
    url = (
        f"https://api.worldbank.org/v2/country/{COUNTRY}/indicator/{code}"
        f"?date={START_YEAR}:{END_YEAR}&format=json&per_page=100"
    )
    
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    
    if not data or len(data) < 2 or not data[1]:
        logger.warning(f"  No data returned for {col_name}")
        return pd.DataFrame(columns=["year", col_name])
    
    rows = []
    for record in data[1]:
        year = record.get("date")
        value = record.get("value")
        if year and value is not None:
            rows.append({"year": int(year), col_name: float(value)})
    
    df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    return df


def fetch_world_bank_data(save_path: str = "data/raw/world_bank_pakistan.csv") -> pd.DataFrame:
    logger.info("Fetching World Bank indicators for Pakistan (REST API)...")
    
    dfs = []
    for code, col_name in INDICATORS.items():
        try:
            df = fetch_indicator(code, col_name)
            if not df.empty:
                dfs.append(df.set_index("year"))
                logger.info(f"  ✓ {col_name} ({len(df)} records)")
            else:
                logger.warning(f"  ✗ {col_name}: empty response")
            time.sleep(0.3)  # polite delay
        except Exception as e:
            logger.warning(f"  ✗ {col_name}: {e}")
    
    if not dfs:
        raise RuntimeError("No data fetched — check internet connection")
    
    combined = pd.concat(dfs, axis=1).reset_index()
    combined.insert(0, "country", "Pakistan")
    combined.insert(1, "country_code", "PK")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    combined.to_csv(save_path, index=False)
    
    logger.info(f"\n✅ Saved {len(combined)} rows x {len(combined.columns)} cols → {save_path}")
    logger.info(f"Missing values per column:\n{combined.isnull().sum().to_string()}")
    return combined


if __name__ == "__main__":
    df = fetch_world_bank_data()
    print("\n--- Preview ---")
    print(df.tail(5).to_string())