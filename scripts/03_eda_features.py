"""
============================================================
LinkedIn Jobs Intelligence System
STEP 3 & 4 — EDA & FEATURE ENGINEERING
============================================================
- Extract skills into a sparse feature matrix
- Encode categorical variables (location, category)
- Create 'seniority_score' and 'skill_count' features
- Visualizes key distributions

Usage:
    python scripts/03_eda_features.py

Output:
    data/processed/jobs_features.csv
    reports/figures/01_skills_frequency.png
    reports/figures/02_salary_distribution.png
============================================================
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# ── Paths ────────────────────────────────────────────────────
CLEANED_CSV   = Path("data/processed/jobs_cleaned.csv")
FEATURES_CSV  = Path("data/processed/jobs_features.csv")
FIGURES_DIR   = Path("reports/figures")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

# ── Feature Engineering ─────────────────────────────────────
def score_seniority(title: str) -> int:
    """Assigns numeric score 1-5 for seniority based on title keywords."""
    title = str(title).lower()
    if any(x in title for x in ['intern', 'junior', 'associate']): return 1
    if any(x in title for x in ['senior', 'sr', 'lead']): return 3
    if any(x in title for x in ['staff', 'principal', 'manager', 'head']): return 4
    if any(x in title for x in ['director', 'vp', 'cto', 'architect']): return 5
    return 2    # Default mid-level

def run_features_eda():
    if not CLEANED_CSV.exists():
        log.error(f"  Missing cleaned data: {CLEANED_CSV}. Run cleaning first.")
        return

    log.info("═" * 55)
    log.info("  LinkedIn Jobs Intelligence — EDA & Feature Engineering")
    log.info("═" * 55)

    df = pd.read_csv(CLEANED_CSV)
    df['seniority_score'] = df['title'].apply(score_seniority)

    # 1. Skill Extraction
    log.info("  Extracting skill features …")
    all_skills = [s.strip() for slist in df['skills'].dropna().str.split(",") for s in slist if s]
    skill_counts = pd.Series(all_skills).value_counts().head(20)

    # 2. EDA: Visuals
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_style("darkgrid")

    # Plot 1: Top 20 Skills
    plt.figure(figsize=(10, 6))
    skill_counts.plot(kind='barh', color='skyblue').invert_yaxis()
    plt.title("🔥 Top 20 In-Demand Skills")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_skills_frequency.png")
    plt.close()

    # Plot 2: Salary distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['salary_numeric'], bins=30, kde=True, color='teal')
    plt.title("💰 Salary Distribution (USD)")
    plt.xlabel("Annual Salary")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_salary_distribution.png")
    plt.close()

    # 3. Categorical Encoding
    le = LabelEncoder()
    df['category_encoded'] = le.fit_transform(df['category'])
    df['skill_count'] = df['skills'].apply(lambda x: len(str(x).split(",")))

    # 4. Sparse Skill Matrix (Dummy encoding for top 20 skills)
    top_skills = skill_counts.index.tolist()
    for skill in top_skills:
        df[f'skill_{skill.lower().replace(" ", "_")}'] = df['skills'].str.contains(skill, case=False, na=False).astype(int)

    # 5. Save Features
    df.to_csv(FEATURES_CSV, index=False)
    log.info(f"\n✅  Feature matrix saved → {FEATURES_CSV}")
    log.info(f"    Features generated: {df.shape[1]}")

if __name__ == "__main__":
    run_features_eda()
