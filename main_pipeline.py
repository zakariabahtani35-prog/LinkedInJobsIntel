"""
============================================================
SR — ENTERPRISE DATA PIPELINE (Main Entry)
============================================================
Orchestrates the LinkedIn Jobs Intelligence System with 
production-grade performance and monitoring.
============================================================
"""

import time
import argparse
from pathlib import Path
from src.utils.logger import get_logger
from core.processing.cleaner import EnterpriseCleaner
from core.quality.profiler import DataProfiler

log = get_logger("enterprise_main")

def run_enterprise_pipeline(raw_path: str, output_path: str = "data/processed/jobs_final"):
    """
    Main pipeline to execute LinkedIn Jobs intelligence cleaning and Profiling.
    
    Args:
        raw_path: Absolute path to the raw input source.
        output_path: Target directory/prefix for results.
    """
    start_time = time.time()
    log.info("[INIT] Initializing LinkedIn Enterprise Data Pipeline...")
    
    try:
        # Step 1: Cleaning and Enrichment Engine
        cleaner = EnterpriseCleaner(output_format="parquet")
        
        # Step 2: Processing and Schema Validation
        report = cleaner.process_file(raw_path, output_path)
        
        # Step 3: Success Telemetry
        elapsed_time = time.time() - start_time
        log.info("=" * 60)
        log.info("[SUCCESS] Pipeline Completed in {elapsed_time:.1f}s")
        log.info(f"Results Persisted: {output_path}.parquet")
        log.info(f"Rows Processed: {report['total_rows']}")
        log.info(f"Duplicates Removed: {report['duplicate_count']}")
        
        # Step 4: Outlier observability
        for col, count in report["outlier_counts"].items():
            if count > 0:
                log.warning(f"[OUTLIER ALERT] Detected {count} extreme values in: {col}")
        log.info("=" * 60)

    except Exception as e:
        log.error(f"[CRITICAL] Pipeline crashed: {e}")
        raise SystemExit(1)

def main():
    """Main CLI handler."""
    parser = argparse.ArgumentParser(description="LinkedIn Enterprise Data Intelligence Pipeline CLI")
    parser.add_argument("--input", required=True, help="Path to raw CSV file.")
    parser.add_argument("--output", default="data/processed/enterprise_jobs", help="Output Parquet path.")
    
    args = parser.parse_args()
    run_enterprise_pipeline(args.input, args.output)

if __name__ == "__main__":
    main()
