"""
============================================================
SR — MODEL TRAINER (PRODUCTION)
============================================================
Implements professional model training, cross-validation,
and serialization for the LinkedIn Jobs Salary Prediction.
============================================================
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from src.utils.logger import get_logger

log = get_logger("model_trainer")

class ModelTrainer:
    def __init__(self, config_path: str = "config/config.json"):
        with open(config_path) as f:
            self.config = json.load(f)
        self.ml_config = self.config["ml"]
        self.paths     = self.config["paths"]
        self.model: Optional[XGBRegressor] = None

    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Handles internal feature selection and preprocessing."""
        df = pd.read_csv(self.paths["feature_matrix"])
        
        # 1. Select numeric feature columns
        skill_cols = [c for c in df.columns if c.startswith("skill_")]
        cat_cols   = ["seniority_score", "skill_count"]
        feature_cols = cat_cols + skill_cols
        
        # 2. Extract X and Y
        X = df[feature_cols].fillna(0).values
        y = df[self.ml_config["target_column"]].fillna(df[self.ml_config["target_column"]].median()).values
        
        return X, y, feature_cols

    def train_salary_model(self) -> Dict[str, Any]:
        """Trains and validates the primary salary regressor."""
        log.info("Starting model training (XGBoost)...")
        
        try:
            X, y, features = self._prepare_data()
            
            # Use configurations from JSON
            props = self.ml_config
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=props["test_size"], random_state=42
            )
            
            log.info(f"Training set: {X_train.shape[0]:,} | Testing: {X_test.shape[0]:,}")
            
            self.model = XGBRegressor(**props["xgb_params"], random_state=42)
            self.model.fit(X_train, y_train)
            
            # Predict and Evaluate
            preds = self.model.predict(X_test)
            mae   = mean_absolute_error(y_test, preds)
            r2    = r2_score(y_test, preds)
            
            log.info(f"Training Finished: R2={r2:.4f} | MAE=${mae:,.0f}")
            
            # Persistence
            model_path = Path(self.paths["model_dir"]) / "salary_model.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)
            
            # Feature Importance Log
            importances = pd.Series(self.model.feature_importances_, index=features).sort_values(ascending=False).head(5)
            log.info(f"Top 5 predictors identified: {importances.index.tolist()}")

            return {
                "mae": mae,
                "r2": r2,
                "path": str(model_path),
                "top_features": importances.to_dict()
            }
            
        except Exception as e:
            log.error(f"Critical training failure: {e}")
            raise
