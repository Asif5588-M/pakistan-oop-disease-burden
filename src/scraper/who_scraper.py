"""
WHO GHO API Scraper — Fixed Version
Uses direct REST API
"""

import requests
import pandas as pd
import os
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WHO_BASE = "https://ghoapi.azureedge.net/api"

WHO_INDICATORS = {
    "NCDMORT3070":                   "ncd_mortality_30_70_pct",
    "SDGPM25":                       "pm25_exposure",
    "WSH_WATER_BASIC":               "basic_water_access_pct",
    "FINPROTECTION_CATA_TOT_10_POP": "catastrophic_health_exp_10pct",
    "FINPROTECTION_CATA_TOT_25_POP": "catastrophic_health_exp_25pct",
}

COUNTRY_CODE = "PAK"


def fetch_who_indicator(indicator: str, col_name: str) -> pd.DataFrame:
    url = f"{WHO_BASE}/{indicator}?$filter=SpatialDim eq '{COUNTRY_CODE}'"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("value", [])

    rows = []
    for row in data:
        year = row.get("TimeDim")
        value = row.get("NumericValue")
        if year and value is not None:
            try:
                rows.append({"year": int(year), col_name: float(value)})
            except:
                pass

    if not rows:
        return pd.DataFrame(columns=["year", col_name])

    df = pd.DataFrame(rows)
    df = df.groupby("year")[col_name].mean().reset_index()
    return df.sort_values("year").reset_index(drop=True)


def fetch_who_data(save_path: str = "data/raw/who_pakistan.csv") -> pd.DataFrame:
    logger.info("Fetching WHO GHO indicators for Pakistan...")
    dfs = []

    for code, col_name in WHO_INDICATORS.items():
        try:
            df = fetch_who_indicator(code, col_name)
            if not df.empty:
                dfs.append(df.set_index("year"))
                logger.info(f"  ✓ {col_name} ({len(df)} records)")
            else:
                logger.warning(f"  ✗ {col_name}: no data")
            time.sleep(0.3)
        except Exception as e:
            logger.warning(f"  ✗ {col_name}: {e}")

    if not dfs:
        raise RuntimeError("No WHO data fetched")

    combined = pd.concat(dfs, axis=1).reset_index()
    combined.insert(0, "country", "Pakistan")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    combined.to_csv(save_path, index=False)
    logger.info(f"\n✅ Saved {len(combined)} rows → {save_path}")
    logger.info(f"Missing values:\n{combined.isnull().sum().to_string()}")
    return combined


if __name__ == "__main__":
    df = fetch_who_data()
    print("\n--- Preview ---")
    print(df.to_string())