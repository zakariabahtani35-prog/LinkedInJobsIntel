"""
============================================================
SR — ENTERPRISE FEATURE ENGINEER (PRO DATA SCIENCE)
============================================================
Handles Multi-label Binarization for skills, role extraction,
temporal features, and log-transformations for ML-readiness.
============================================================
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from src.utils.logger import get_logger

log = get_logger("feature_engineer")

class FeatureEngineer:
    """Professional data science interface for LinkedIn job feature extraction."""
    
    def __init__(self, data_path: str, output_dir: str = "data/processed/features"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = Path("models/transformers")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Data
        if self.data_path.suffix == ".parquet":
            self.df = pd.read_parquet(self.data_path)
        else:
            self.df = pd.read_csv(self.data_path)
            
        log.info(f"Loaded {len(self.df)} records for feature transformation.")

    # ── 1. Skill Extraction & Binarization ───────────────────
    def extract_skill_vectors(self, top_n: int = 25) -> pd.DataFrame:
        """Applies MultiLabelBinarizer to convert skills list into ML flags."""
        log.info(f"Extracting binarized skill vectors (top {top_n})...")
        
        # 1. Clean list of skills per row
        skill_lists = self.df["skills"].fillna("").str.split(", ").tolist()
        skill_lists = [[s for s in l if s] for l in skill_lists]
        
        # 2. Fit MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        binary_matrix = mlb.fit_transform(skill_lists)
        skill_df = pd.DataFrame(binary_matrix, columns=[f"skill_{c.lower().replace(' ', '_')}" for c in mlb.classes_])
        
        # 3. Filter only top columns to prevent sparsity explosion
        top_skill_cols = skill_df.sum().sort_values(ascending=False).head(top_n).index
        skill_df = skill_df[top_skill_cols]
        
        # 4. Save metadata
        with open(self.artifacts_dir / "mlb_classes.pkl", "wb") as f:
            pickle.dump(list(top_skill_cols), f)
            
        # 5. Add skill count feature
        self.df["skill_count"] = self.df["skills"].fillna("").apply(lambda x: len(x.split(",")) if x else 0)
        
        log.info(f"Generated {len(top_skill_cols)} skill features.")
        return skill_df

    # ── 2. Experience & Role Mapping ──────────────────────────
    def extract_role_experience(self) -> pd.DataFrame:
        """Transforms text titles into structured Role and Seniority features."""
        log.info("Mapping role and seniority hierarchies...")
        
        # Seniority Numeric (Junior=0, Mid=1, Senior=2)
        def map_seniority(title: str) -> int:
            t = str(title).lower()
            if any(x in t for x in ["junior", "intern", "jr", "trainee"]): return 0
            if any(x in t for x in ["senior", "sr", "lead", "staff", "principal"]): return 2
            return 1 # Default Mid
            
        self.df["seniority_numeric"] = self.df["title"].apply(map_seniority)
        
        # Role/Domain Extraction
        role_map = {
            "Data Engineer": ["data engineer"],
            "Data Scientist": ["data scientist", "machine learning", "ml"],
            "Analyst": ["analyst", "analytics", "bi"],
            "Software": ["developer", "software", "backend", "frontend", "fullstack"],
            "Cloud/DevOps": ["devops", "cloud", "aws", "azure", "infrastructure"]
        }
        
        def map_role(title: str) -> str:
            t = str(title).lower()
            for role, keys in role_map.items():
                if any(x in t for x in keys): return role
            return "General Tech"
            
        self.df["role_broad"] = self.df["title"].apply(map_role)
        
        # One-hot encode the extracted roles
        role_dummies = pd.get_dummies(self.df["role_broad"], prefix="role")
        return role_dummies

    # ── 3. Salary Transformations ────────────────────────────
    def process_compensation(self):
        """Normalizes compensation metrics and adds derived ratio features."""
        log.info("Processing salary normalization and feature ratios...")
        
        # 1. Log-transform (Stabilize variance for skewness)
        # Avoid log(0) by using offset
        self.df["salary_log"] = np.log1p(self.df["salary_numeric"])
        
        # 2. Ratio features (Product-level insights)
        # median_sal_per_skill = (Salary/SkillCount)
        self.df["salary_skill_ratio"] = self.df["salary_numeric"] / (self.df["skill_count"] + 1)
        
        # 3. Min-Max Scaling for numeric features (ML prep)
        scaler = MinMaxScaler()
        self.df[["seniority_norm", "skill_count_norm"]] = scaler.fit_transform(
            self.df[["seniority_numeric", "skill_count"]]
        )

    # ── 4. Temporal Features ─────────────────────────────────
    def generate_temporal_features(self):
        """Extracts periodicity and freshness from job timestamps."""
        log.info("Generating temporal trend features...")
        
        self.df["published_at"] = pd.to_datetime(self.df["published_at"], errors='coerce')
        # Extract Month/Weekday
        self.df["post_month"]   = self.df["published_at"].dt.month.fillna(0).astype(int)
        self.df["post_weekday"] = self.df["published_at"].dt.weekday.fillna(0).astype(int)
        
        # Age of posting (delta)
        now = pd.Timestamp.now()
        self.df["job_age_days"] = (now - self.df["published_at"]).dt.days.fillna(0)

    # ── 5. Master Orchestrator ───────────────────────────────
    def build_feature_set(self) -> pd.DataFrame:
        """Runs the complete transformation pipeline and saves the result."""
        log.info("BUILDING MASTER FEATURE SET...")
        
        # Component generation
        skill_df  = self.extract_skill_vectors(top_n=25)
        role_df   = self.extract_role_experience()
        self.process_compensation()
        self.generate_temporal_features()
        
        # Cleanup location
        loc_dummies = pd.get_dummies(self.df["location"], prefix="loc")
        
        # Combine everything
        final_df = pd.concat([
            self.df[[
                "id", "salary_numeric", "salary_log", "seniority_numeric", 
                "skill_count", "job_age_days", "salary_skill_ratio",
                "post_month", "post_weekday"
            ]],
            skill_df,
            role_df,
            loc_dummies
        ], axis=1)
        
        # Final persistence
        out_parquet = self.output_dir / "ml_ready_features.parquet"
        final_df.to_parquet(out_parquet, compression="snappy")
        
        log.info(f"FINISHED: Feature matrix build successful ({final_df.shape[0]}x{final_df.shape[1]})")
        log.info(f"Feature set stored: {out_parquet}")
        return final_df
