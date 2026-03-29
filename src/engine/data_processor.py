"""
============================================================
SR — DATA PROCESSOR (PRODUCTION)
============================================================
Implements industrial-grade data cleaning, normalization,
and feature engineering as a maintainable class.
============================================================
"""

import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from src.utils.logger import get_logger

log = get_logger("data_processor")

class DataProcessor:
    def __init__(self, config_path: str = "config/config.json"):
        with open(config_path) as f:
            self.config = json.load(f)
        self.paths = self.config["paths"]
        self.df: Optional[pd.DataFrame] = None

    # ── Cleaning Logic ──────────────────────────────────────────
    def parse_salary(self, salary_str: str) -> float:
        """Parses annual USD from messy strings with robust error handling."""
        if not isinstance(salary_str, str) or not salary_str:
            return np.nan
        
        try:
            val = salary_str.lower().replace(",", "")
            # Regex for all numeric groups
            nums = re.findall(r'(\d+[\d\.]*)', val)
            if not nums: return np.nan
            
            floats = [float(n) for n in nums]
            mid_val = sum(floats) / len(floats)
            
            # Application of multipliers
            if 'k' in val: mid_val *= 1000
            
            # Annualization logic
            if mid_val < 500: mid_val *= 160 * 12  # Hourly to Annual
            elif mid_val < 15000: mid_val *= 12    # Monthly to Annual
            
            return mid_val if 15000 < mid_val < 800000 else np.nan
        except (ValueError, ZeroDivisionError):
            return np.nan

    def normalize_skills(self, tags: str) -> str:
        """Standardizes skill taxonomies using a unified map."""
        if not isinstance(tags, str): return ""
        
        skill_map = {
            r'python\d*': 'Python',
            r'js|javascript|node': 'JavaScript',
            r'react': 'React',
            r'aws|amazon': 'AWS',
            r'postgres|sql|database': 'SQL'
        }
        
        normalized = []
        raw_skills = [t.strip().lower() for t in tags.split(",") if t.strip()]
        
        for s in raw_skills:
            matched = False
            for pattern, canonical in skill_map.items():
                if re.search(pattern, s):
                    normalized.append(canonical)
                    matched = True
                    break
            if not matched:
                normalized.append(s.title())
        
        return ", ".join(sorted(list(set(normalized))))

    # ── Pipeline Orchestration ──────────────────────────────────
    def run_cleaning_pipeline(self) -> pd.DataFrame:
        """Main entry point for the cleaning stage."""
        log.info("Starting production cleaning pipeline...")
        
        try:
            raw_path = Path(self.paths["raw_data"])
            if not raw_path.exists():
                raise FileNotFoundError(f"Missing raw data input: {raw_path}")

            df = pd.read_csv(raw_path)
            log.info(f"Read {len(df):,} raw records.")

            # Pipeline sequence
            df = df.drop_duplicates(subset=["id"])
            df["salary_numeric"] = df["salary"].apply(self.parse_salary)
            df["skills"] = df["tags"].apply(self.normalize_skills)
            
            # Handle empty values with business-default logic
            df["salary_numeric"] = df["salary_numeric"].fillna(df["salary_numeric"].median())
            df["category"] = df["category"].fillna("General")
            
            processed_path = Path(self.paths["processed_data"])
            processed_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(processed_path, index=False)
            
            self.df = df
            log.info(f"Cleaning complete. Output: {processed_path}")
            return df
        except Exception as e:
            log.error(f"Critical failure in cleaning pipeline: {e}")
            raise

    def run_feature_engineering(self) -> pd.DataFrame:
        """Generates ML-ready features and calculates distributions."""
        if self.df is None:
            self.df = pd.read_csv(self.paths["processed_data"])
            
        log.info("Generating production feature vectors...")
        
        # 1. Seniority scoring (Scale 1-5)
        keywords = {
            'intern': 1, 'junior': 1, 'lead': 4, 'staff': 4, 
            'principal': 5, 'director': 5, 'cto': 5
        }
        self.df['seniority_score'] = self.df['title'].str.lower().apply(
            lambda x: next((v for k, v in keywords.items() if k in x), 2)
        )
        
        # 2. Skill count
        self.df['skill_count'] = self.df['skills'].fillna("").apply(lambda x: len(x.split(",")) if x else 0)
        
        # 3. Sparse skills Matrix (Top 15)
        top_skills = self.df['skills'].str.split(", ").explode().value_counts().head(15).index
        for skill in top_skills:
            col_name = f"skill_{skill.replace(' ', '_').lower()}"
            self.df[col_name] = self.df['skills'].str.contains(skill, case=False, na=False).astype(int)
            
        feature_path = Path(self.paths["feature_matrix"])
        self.df.to_csv(feature_path, index=False)
        log.info(f"Features generated. Output: {feature_path}")
        return self.df
