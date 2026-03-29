"""
============================================================
SR — PRODUCTION PIPELINE ORCHESTRATOR
============================================================
The master runner using industrial-grade logic and logging.
============================================================
"""

import time
import json
import logging
from pathlib import Path
from src.utils.logger import get_logger
from src.engine.data_processor import DataProcessor
from src.engine.model_trainer import ModelTrainer

# Application log
log = get_logger("orchestrator")

def run_main_pipeline():
    """Executes the end-to-end data intelligence process."""
    start_time = time.time()
    log.info("🚀 [SYSTEM INIT] Starting production LinkedIn Jobs Intelligence pipeline...")
    
    try:
        # Load Config
        config_path = "config/config.json"
        
        # 1. Processing Stage
        processor = DataProcessor(config_path)
        processor.run_cleaning_pipeline()
        processor.run_feature_engineering()
        
        # 2. Training Stage
        trainer = ModelTrainer(config_path)
        result = trainer.train_salary_model()
        
        # Finish
        elapsed = time.time() - start_time
        log.info(f"✅ [SUCCESS] Total Pipeline Time: {elapsed:.1f}s")
        log.info(f"📊 Final Model Metric: R2={result['r2']:.4f} | MAE=${result['mae']:,.0f}")
        
    except Exception as e:
        log.error(f"❌ [CRITICAL] Pipeline reached unrecoverable state: {e}")
        raise
    
if __name__ == "__main__":
    run_main_pipeline()
