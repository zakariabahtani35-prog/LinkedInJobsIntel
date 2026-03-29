"""
============================================================
SR — ENTERPRISE MODEL TRAINER (XGBOOST ENGINE)
============================================================
Handles cross-validated training, hyperparameter tuning,
and production serialization for job salary prediction.
============================================================
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple, List
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.utils.logger import get_logger

log = get_logger("model_trainer")

class SalaryModelTrainer:
    """Professional ML interface for LinkedIn Salary Prediction."""
    
    def __init__(self, data_path: str, model_dir: str = "models", report_dir: str = "reports/models"):
        self.data_path = Path(data_path)
        self.model_dir = Path(model_dir)
        self.report_dir = Path(report_dir)
        
        # Ensure directories exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Engineered Data
        self.df = pd.read_parquet(self.data_path)
        log.info(f"Loaded {len(self.df)} records for ML training.")

    # ── 1. Data Splitting & Prep ─────────────────────────────
    def prepare_datasets(self, target_col: str = "salary_log") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Splits the feature matrix into repeatable training and test sets."""
        log.info(f"Preparing train/test split (target={target_col})...")
        
        # Drop raw/redundant columns for training
        # Ensure target is valid (Drop NaNs/Infs)
        mask = self.df[target_col].notnull() & np.isfinite(self.df[target_col])
        clean_df = self.df[mask]
        
        X = clean_df.drop(columns=["id", "salary_numeric", "salary_log", "salary_skill_ratio"], errors='ignore')
        y = clean_df[target_col]
        
        # 70/30 Initial Split (then further split for validation)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42
        )
        
        log.info(f"Matrix shapes: Train {X_train.shape}, Test {X_test.shape}")
        return X_train, X_test, y_train, y_test

    # ── 2. Training Pipeline with Tuning ──────────────────────
    def train_with_tuning(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """Executes a GridSearchCV over an XGBoost Pipeline for optimal performance."""
        log.info("Initializing XGBoost Training Pipeline with GridSearch...")
        
        # Define Pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", XGBRegressor(random_state=42, objective="reg:squarederror"))
        ])
        
        # Simple but effective Grid for low-latency tuning
        param_grid = {
            "regressor__n_estimators": [100, 200, 500],
            "regressor__max_depth": [3, 5, 7],
            "regressor__learning_rate": [0.01, 0.05, 0.1]
        }
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=1
        )
        
        log.info("Executing 5-fold cross-validation...")
        grid_search.fit(X_train, y_train)
        
        log.info(f"Best CV Score: {grid_search.best_score_:.4f}")
        log.info(f"Optimal Hyperparams: {grid_search.best_params_}")
        
        return grid_search.best_estimator_

    # ── 3. Evaluation & Reporting ───────────────────────────
    def evaluate_model(self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
        """Generates residuals and calculates regression metrics (R2, RMSE, MAE)."""
        log.info("Calculating production performance metrics...")
        
        # Predictions (In Log Space)
        preds_log = model.predict(X_test)
        
        # Back-transform to USD for real-world metrics
        y_test_real = np.expm1(y_test)
        preds_real  = np.expm1(preds_log)
        
        # Metrics
        results = {
            "R2_Score": r2_score(y_test, preds_log),
            "MAE_USD": mean_absolute_error(y_test_real, preds_real),
            "RMSE_USD": root_mean_squared_error(y_test_real, preds_real)
        }
        
        # Save Report
        with open(self.report_dir / "performance.txt", "w") as f:
            f.write("=== XGBOOST SALARY MODEL EVALUATION ===\n")
            for k, v in results.items():
                f.write(f"{k}: {v:.4f}\n")
                log.info(f"Metric: {k} = {v:.4f}")
                
        # Generate Residual Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test_real, preds_real, alpha=0.5, color="#00CC96")
        plt.plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], 'r--')
        plt.xlabel("Actual Salary ($)")
        plt.ylabel("Predicted Salary ($)")
        plt.title("Actual vs Predicted Salaries (Test Set)")
        plt.savefig(self.report_dir / "residual_plot.png")
        plt.close()

    # ── 4. Feature Importance Extraction ────────────────────
    def export_importance(self, model: Pipeline, feature_names: List[str]):
        """Saves the top predictors to CSV for stakeholder insight."""
        log.info("Extracting feature significance vectors...")
        
        importance = model.named_steps["regressor"].feature_importances_
        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(ascending=False, by="Importance")
        
        imp_df.to_csv(self.report_dir / "feature_importance.csv", index=False)
        log.info(f"Top 3 Predictors: {imp_df['Feature'].iloc[:3].tolist()}")

    # ── 5. Main Orchestration ──────────────────────────────
    def run_production_training(self):
        """Complete ML workflow: Split, Tune, Eval, and Serialize."""
        log.info("STARTING PRODUCTION MODEL TRAINING...")
        
        # 1. Split
        X_train, X_test, y_train, y_test = self.prepare_datasets()
        
        # 2. Train/Tune
        best_model = self.train_with_tuning(X_train, y_train)
        
        # 3. Eval
        self.evaluate_model(best_model, X_test, y_test)
        
        # 4. Importance
        feature_names = X_train.columns.tolist()
        self.export_importance(best_model, feature_names)
        
        # 5. Serialize Model
        model_file = self.model_dir / "xgb_salary_model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(best_model, f)
        
        # 6. Save Feature Schema (CRITICAL for inference alignment)
        schema_path = Path("models/transformers/feature_schema.pkl")
        with open(schema_path, "wb") as f:
            pickle.dump(feature_names, f)
        log.info(f"Feature schema persisted: {schema_path} ({len(feature_names)} features)")
            
        log.info(f"SUCCESS: Model serialized to {model_file}")
