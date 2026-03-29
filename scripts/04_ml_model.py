"""
============================================================
LinkedIn Jobs Intelligence System
STEP 5 — MACHINE LEARNING: Salary prediction model
============================================================
- Trains an XGBoost Regressor to predict annual salary (USD)
- Evaluates model using MAE, RMSE, and R2
- Interprets feature importance for actionable business insights

Usage:
    python scripts/04_ml_model.py

Output:
    models/salary_model.pkl
    reports/model_results.txt
============================================================
"""

import logging
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ────────────────────────────────────────────────────
FEATURES_CSV  = Path("data/processed/jobs_features.csv")
MODEL_FILE    = Path("models/salary_model.pkl")
REPORT_FILE   = Path("reports/model_results.txt")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

# ── ML Workflow ─────────────────────────────────────────────
def run_model():
    if not FEATURES_CSV.exists():
        log.error(f"  Missing features data: {FEATURES_CSV}. Run feature engineering first.")
        return

    log.info("═" * 55)
    log.info("  LinkedIn Jobs Intelligence — Machine Learning")
    log.info("═" * 55)

    df = pd.read_csv(FEATURES_CSV)

    # 1. Feature selection
    skill_cols = [c for c in df.columns if c.startswith("skill_")]
    X_cols     = ["seniority_score", "skill_count", "category_encoded"] + skill_cols
    X          = df[X_cols].fillna(0).values
    y          = df['salary_numeric'].fillna(df['salary_numeric'].median()).values

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Model Training
    log.info("  Training XGBoost regressor …")
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluation
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)

    log.info(f"  Model Result: MAE=${mae:,.0f} | RMSE=${rmse:,.0f} | R²={r2:.3f}")

    # 5. Interpretation
    importances = pd.Series(model.feature_importances_, index=X_cols).sort_values(ascending=False)
    log.info("\n  Top Feature Importances:")
    for feat, imp in importances.head(5).items():
        log.info(f"    - {feat:20}: {imp:.3f}")

    # 6. Save Model
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    # 7. Write Report
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_FILE.open("w") as f:
        f.write("=== ML MODEL REPORT: SALARY PREDICTION ===\n")
        f.write(f"Algorithm: XGBoost Regressor\n")
        f.write(f"MAE: ${mae:,.0f}\n")
        f.write(f"RMSE: ${rmse:,.0f}\n")
        f.write(f"R2 Score: {r2:.4f}\n\n")
        f.write("--- Top 5 Predictors ---\n")
        for feat, imp in importances.head(5).items():
            f.write(f"{feat}: {imp:.4f}\n")

    log.info(f"\n✅  Model results & serialized artifact saved → {MODEL_FILE}")

if __name__ == "__main__":
    run_model()
