"""
============================================================
LinkedIn Jobs Intelligence System
STEP 5 — SALARY PREDICTION (XGBOOST PRODUCTION)
============================================================
Usage:
    py scripts/05_train_salary_model.py
============================================================
"""

import os
import sys
sys.path.append(os.getcwd())

from core.models.trainer import SalaryModelTrainer
from src.utils.logger import get_logger

log = get_logger("script_05")

def main():
    try:
        input_data = "data/processed/features/ml_ready_features.parquet"
        if not os.path.exists(input_data):
            log.error(f"Missing required input: {input_data}. Run Step 4 first.")
            return

        log.info("Executing Script 05: Salary Model Training & Evaluation")
        
        # 1. Initialize Pro Trainer
        trainer = SalaryModelTrainer(input_data)
        
        # 2. Complete ML Workflow (Split, Tune, Eval, Save)
        trainer.run_production_training()
        
        log.info("=" * 60)
        log.info("SUCCESS: Salary Prediction Model Optimized & Saved")
        log.info(f"Model: models/xgb_salary_model.pkl")
        log.info(f"Performance: reports/models/performance.txt")
        log.info(f"Insights: reports/models/feature_importance.csv")
        log.info("=" * 60)
        
    except Exception as e:
        log.error(f"Script 05 Error: {e}")

if __name__ == "__main__":
    main()
