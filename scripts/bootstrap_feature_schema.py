"""
============================================================
BOOTSTRAP — Generate feature_schema.pkl from existing model
============================================================
Run once if you trained the model BEFORE trainer.py was
patched to save feature_schema.pkl automatically.

Usage:
    py scripts/bootstrap_feature_schema.py

What it does:
  1. Loads the existing XGBoost Pipeline
  2. Extracts the feature count from the trained XGBRegressor
  3. Reconstructs the column order from feature_importance.csv
     (which was saved by the same training run)
  4. Persists models/transformers/feature_schema.pkl
============================================================
"""

import pickle
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────
model_path    = Path("models/xgb_salary_model.pkl")
imp_path      = Path("reports/models/feature_importance.csv")
schema_path   = Path("models/transformers/feature_schema.pkl")

schema_path.parent.mkdir(parents=True, exist_ok=True)

# ── 1. Load model ─────────────────────────────────────────
print("Loading model...")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# The sklearn Pipeline exposes the XGBRegressor via named_steps
xgb = model.named_steps["regressor"]
n_features = xgb.n_features_in_
print(f"Model trained on {n_features} features.")

# ── 2. Reconstruct schema from feature_importance.csv ─────
# feature_importance.csv lists features in model order (same
# order as X_train.columns used during trainer.export_importance)
print("Reading feature importance CSV...")
imp_df = pd.read_csv(imp_path)
schema = imp_df["Feature"].tolist()

# Sanity check
if len(schema) != n_features:
    print(
        f"⚠️  WARNING: feature_importance.csv has {len(schema)} rows but "
        f"model expects {n_features} features. "
        f"Schema may be incomplete — re-run training for authoritative schema."
    )
else:
    print(f"✅ Schema validated: {len(schema)} features match model expectation.")

# ── 3. Persist ────────────────────────────────────────────
with open(schema_path, "wb") as f:
    pickle.dump(schema, f)

print(f"✅ Feature schema saved to: {schema_path}")
print("Features:", schema)
