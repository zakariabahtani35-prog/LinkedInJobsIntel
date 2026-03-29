"""
============================================================
LinkedIn Jobs Intelligence System
STEP 3 — PROFESSIONAL EDA (MASTER SCRIPT)
============================================================
Usage:
    py scripts/03_pro_eda.py
============================================================
"""

import os
import sys
sys.path.append(os.getcwd())
from core.analysis.eda_engine import EDAEngine
from src.utils.logger import get_logger

log = get_logger("script_03")

def main():
    try:
        data_path = "data/processed/enterprise_jobs.parquet"
        if not os.path.exists(data_path):
            log.warning("Parquet fallback: data/processed/enterprise_jobs.csv")
            data_path = "data/processed/enterprise_jobs.csv"
            
        log.info(f"Executing Script 03: Professional EDA on {data_path}")
        
        # 1. Initialize Pro EDA Engine
        engine = EDAEngine(data_path)
        
        # 2. Automated Exploration
        engine.generate_summary()
        engine.run_univariate()
        engine.run_bivariate()
        
        # 3. Dynamic High-Level Strategic Analysis
        insights = engine.extract_strategic_insights()
        
        log.info("=" * 60)
        log.info("SUCCESS: Professional EDA Reports Generated")
        log.info(f"Dashboard Visuals: reports/eda/figures/")
        log.info(f"Key Strategy Insight: {insights[0]}")
        log.info("=" * 60)
        
    except Exception as e:
        log.error(f"Script 03 Error: {e}")

if __name__ == "__main__":
    main()
