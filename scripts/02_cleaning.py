"""
============================================================
LinkedIn Jobs Intelligence System
STEP 2 — DATA CLEANING: Pipeline
============================================================
- Handles missing values
- Normalizes skill names
- Converts salary ranges into numeric formats (annual USD)
- Removes duplicates
- Logs all transformations for audit

Usage:
    python scripts/02_cleaning.py

Output:
    data/processed/jobs_cleaned.csv
============================================================
"""

import re
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
RAW_CSV       = Path("data/raw/jobs_raw.csv")
CLEANED_CSV   = Path("data/processed/jobs_cleaned.csv")
LOG_PATH      = Path("reports/cleaning_log.txt")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

# ── Salary Parser ──────────────────────────────────────────
def parse_salary(val: str) -> float:
    """Intelligent parsing of salary strings to numeric annual USD."""
    if not isinstance(val, str) or not val: return np.nan

    # 1. Standardize
    val = val.lower().replace(",", "")

    # 2. Extract numbers
    nums = re.findall(r'(\d+[\d\.]*)', val)
    if not nums: return np.nan
    nums = [float(n) for n in nums]

    # 3. Handle mid-point
    res = sum(nums) / len(nums)

    # 4. Handle 'k' / multiplier
    if 'k' in val: res *= 1000

    # 5. Annualize (assuming Remotive/LinkedIn context)
    if res < 500: res *= 160 * 12   # hourly to monthly/annual
    elif res < 15000: res *= 12     # monthly to annual

    return res if 15000 < res < 800000 else np.nan

# ── Skill Normalizer ───────────────────────────────────────
SKILL_MAP = {
    r'python\d*': 'Python',
    r'java\s*script|js|node': 'JavaScript',
    r'rect|react\.js': 'React',
    r'aws|amazon web services': 'AWS',
    r'sql|postgres|mysql': 'SQL'
}

def normalize_skills(tags: str) -> str:
    """Normalize skill strings from tags."""
    if not isinstance(tags, str): return ""
    skills = [t.strip().lower() for t in tags.split(",")]
    normalized = []
    for s in skills:
        found = False
        for pattern, replacement in SKILL_MAP.items():
            if re.search(pattern, s):
                normalized.append(replacement)
                found = True
                break
        if not found:
            normalized.append(s.title())
    return ", ".join(sorted(list(set(normalized))))

# ── Cleaner ────────────────────────────────────────────────
def run_cleaning():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not RAW_CSV.exists():
        log.error(f"  Missing raw data: {RAW_CSV}. Run scraper first.")
        return

    log.info("═" * 55)
    log.info("  LinkedIn Jobs Intelligence — Data Cleaning")
    log.info("═" * 55)

    df = pd.read_csv(RAW_CSV)
    orig_count = len(df)
    log.info(f"  Initial records: {orig_count:,}")

    # 1. Drop duplicates
    df.drop_duplicates(subset=["id"], inplace=True)
    log.info(f"  Duplicates removed: {orig_count - len(df):,}")

    # 2. Parse Salary
    log.info("  Parsing salaries …")
    df['salary_numeric'] = df['salary'].apply(parse_salary)
    sal_found = df['salary_numeric'].notna().sum()
    log.info(f"  Valid salaries extracted: {sal_found:,} ({sal_found/len(df)*100:.1f}%)")

    # 3. Normalize Skills
    log.info("  Normalizing skills mapping …")
    df['skills'] = df['tags'].apply(normalize_skills)

    # 4. Missing values
    df['category'] = df['category'].fillna("General")
    df['location'] = df['location'].fillna("Worldwide")
    df['salary_numeric'] = df['salary_numeric'].fillna(df['salary_numeric'].median())

    # 5. Output
    CLEANED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_CSV, index=False)
    log.info(f"\n✅  Cleaned data saved → {CLEANED_CSV}")

if __name__ == "__main__":
    run_cleaning()
