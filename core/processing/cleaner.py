"""
============================================================
ENTERPRISE — DATA CLEANER (SR DATA ENGINEER)
============================================================
Handles performance-optimized data cleaning with integrated
profiling and Parquet output.
============================================================
"""

import re
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path
from core.quality.profiler import DataProfiler
from src.utils.logger import get_logger

log = get_logger("enterprise_cleaner")

class EnterpriseCleaner:
    """A production-grade cleaner focusing on performance and observability."""

    EXPECTED_COLUMNS = ["id", "title", "company", "category", "tags", "salary", "location", "published_at"]

    def __init__(self, output_format: str = "parquet"):
        """
        Initializes the Cleaner Engine.
        
        Args:
            output_format: Filesystem format ('parquet' or 'csv'). Parquet is default for Enterprise logic.
        """
        self.output_format = output_format.lower()
        self.profiler = DataProfiler()

    def process_file(self, input_path: str, output_path: str):
        """
        Loads raw data, performs validation, cleans, and persists results.
        
        Args:
            input_path: Absolute path to the raw input.
            output_path: Absolute path to the destination (no extension required).
        """
        try:
            log.info(f"Processing started for: {input_path}")
            
            # 1. Performance-optimized Load (read_csv with specified subset)
            df = pd.read_csv(input_path, usecols=self.EXPECTED_COLUMNS)
            
            # 2. Contract Validation
            self.profiler.validate_schema(df, self.EXPECTED_COLUMNS)
            
            # 3. Vectorized Cleaning (Avoiding loops)
            df = self._vectorized_clean(df)
            
            # 4. Persistence
            final_path = self._save_data(df, output_path)
            
            # 5. Post-Process Quality Profiling
            report = self.profiler.generate_quality_report(df, "cleaned_layer")
            log.info(f"Step Successful: {final_path}")
            return report
            
        except Exception as e:
            log.error(f"Unrecoverable failure in cleaner: {e}")
            raise

    def _vectorized_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies vectorized pandas operations instead of O(n) loops."""
        df = df.drop_duplicates(subset=["id"]).copy()
        
        # Fast normalization
        df["skills"]   = df["tags"].str.lower().str.replace(r'[^a-zA-Z0-9,\s]', '', regex=True)
        df["category"] = df["category"].fillna("N/A").str.strip()
        
        # Numeric salary prep
        # We process thousands of rows instantly via vectorized string method
        df["salary_numeric"] = (
            df["salary"]
            .str.lower()
            .str.replace(",", "")
            .str.extract(r'(\d+)')[0]
            .astype(float)
        )
        
        # Scaling annual logic based on thresholds
        # Using np.select or Masking for performance
        df.loc[df["salary_numeric"] < 500, "salary_numeric"] *= (160 * 12)
        df.loc[(df["salary_numeric"] >= 500) & (df["salary_numeric"] < 15000), "salary_numeric"] *= 12
        
        # Handle outliers (anything outside $20k-$500k is capped or Null in Enterprise context)
        df.loc[df["salary_numeric"] > 600000, "salary_numeric"] = np.nan
        df.loc[df["salary_numeric"] < 20000, "salary_numeric"]  = np.nan
        df["salary_numeric"] = df["salary_numeric"].fillna(df["salary_numeric"].median())
        
        return df

    def _save_data(self, df: pd.DataFrame, path_prefix: str) -> str:
        """Handles multi-format persistence logic."""
        Path(path_prefix).parent.mkdir(parents=True, exist_ok=True)
        
        if self.output_format == "parquet":
            out = f"{path_prefix}.parquet"
            df.to_parquet(out, compression="snappy", engine="pyarrow")
        else:
            out = f"{path_prefix}.csv"
            df.to_csv(out, index=False)
            
        return out
