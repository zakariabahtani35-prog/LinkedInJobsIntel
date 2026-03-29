"""
============================================================
LinkedIn Jobs Intelligence System
STEP 4 — PRODUCTION FEATURE ENGINEERING
============================================================
Transforms cleaned data into ML-ready inputs.
- One-hot / MLB Encoding
- Role Extraction
- Salary Log Normalization
- Temporal Trending

Usage:
    py scripts/04_feature_engineering.py
============================================================
"""

import os
import sys
sys.path.append(os.getcwd())

from core.processing.feature_engineer import FeatureEngineer
from src.utils.logger import get_logger

log = get_logger("script_04")

def main():
    try:
        input_data = "data/processed/enterprise_jobs.parquet"
        if not os.path.exists(input_data):
            log.error(f"Missing required input: {input_data}. Run Step 1 & 2 first.")
            return

        log.info("Executing Script 04: Production Feature Engineering")
        
        # 1. Initialize Pro Feature Engineering Engine
        fe = FeatureEngineer(input_data)
        
        # 2. Extract Master Feature Set
        features_df = fe.build_feature_set()
        
        log.info("=" * 60)
        log.info("SUCCESS: ML-Ready Feature Matrix Generated")
        log.info(f"Matrix Dimensions: {features_df.shape}")
        log.info(f"File: data/processed/features/ml_ready_features.parquet")
        log.info(f"Transformers saved: models/transformers/")
        log.info("=" * 60)
        
    except Exception as e:
        log.error(f"Script 04 Error: {e}")

if __name__ == "__main__":
    main()
