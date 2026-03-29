import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.utils.logger import get_logger

log = get_logger("data_profiler")

class DataProfiler:
    """Enterprise-level Data Quality check layer."""
    
    @staticmethod
    def generate_quality_report(df: pd.DataFrame, dataset_name: str = "main") -> Dict[str, Any]:
        """Runs validation checks and generates a structured report."""
        log.info(f"Generating Data Quality Report for: {dataset_name}")
        
        report = {
            "dataset": dataset_name,
            "total_rows": len(df),
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_count": int(df.duplicated().sum()),
            "numeric_stats": df.describe().to_dict()
        }
        
        # Outlier Detection (Z-score > 3)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        for col in numeric_cols:
            z_scores = (df[col] - df[col].mean()) / df[col].std()
            outlier_count = (np.abs(z_scores) > 3).sum()
            outliers[col] = int(outlier_count)
        
        report["outlier_counts"] = outliers
        
        log.info(f"Quality Check Finished: {report['duplicate_count']} duplicates found.")
        return report

    @staticmethod
    def validate_schema(df: pd.DataFrame, expected_cols: List[str]):
        """Ensures the incoming data matches the required contract."""
        missing = set(expected_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Schema Validation Failed: Missing columns {missing}")
        log.info("Schema Validation: Passed")
