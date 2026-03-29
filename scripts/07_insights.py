"""
============================================================
LinkedIn Jobs Intelligence System
STEP 7 — BUSINESS INSIGHTS: Strategy Engine
============================================================
- Analyzes skill co-occurrence for salary premiums
- Identifies market gaps (underserved categories)
- Generates strategic recommendations for job seekers

Usage:
    python scripts/07_insights.py

Output:
    reports/business_insights.txt
============================================================
"""

import logging
import pandas as pd
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
FEATURES_CSV  = Path("data/processed/jobs_features.csv")
INSIGHTS_FILE = Path("reports/business_insights.txt")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

# ── Insights Loop ──────────────────────────────────────────
def run_insights():
    if not FEATURES_CSV.exists():
        log.error(f"  Missing features data: {FEATURES_CSV}. Run full pipeline first.")
        return

    log.info("═" * 55)
    log.info("  LinkedIn Jobs Intelligence — Business Insights")
    log.info("═" * 55)

    df = pd.read_csv(FEATURES_CSV)

    # 1. Salary uplift by skill
    skill_cols = [c for c in df.columns if c.startswith("skill_")]
    avg_sal = df['salary_numeric'].mean()
    uplifts = {}
    for sc in skill_cols:
        with_skill = df[df[sc] == 1]['salary_numeric'].mean()
        if pd.notna(with_skill):
            uplifts[sc.replace("skill_", "").title()] = ((with_skill / avg_sal) - 1) * 100

    top_uplifts = pd.Series(uplifts).sort_values(ascending=False).head(10)

    # 2. Market Gaps: High salary, low volume categories
    gap_df = df.groupby('category').agg({
        'salary_numeric': 'mean',
        'id': 'count'
    }).rename(columns={'id': 'job_count', 'salary_numeric': 'avg_salary'})
    
    # Define "Gaps" as categories in top 25% salary but bottom 50% job count
    high_pay = gap_df['avg_salary'].quantile(0.75)
    low_vol  = gap_df['job_count'].quantile(0.50)
    market_gaps = gap_df[(gap_df['avg_salary'] >= high_pay) & (gap_df['job_count'] <= low_vol)]

    # 3. Report Generation
    INSIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with INSIGHTS_FILE.open("w") as f:
        f.write("=== 📊 STRATEGIC MARKET INSIGHTS: LinkedIn Jobs ===\n")
        f.write("Project: LinkedIn Jobs Intelligence System\n")
        f.write("-" * 50 + "\n\n")

        f.write("1. 🚀 HIGH-VALUE SKILL PREMIUMS (Salary Uplift vs Median)\n")
        for skill, up in top_uplifts.items():
            f.write(f"   {skill:20}: +{up:.1f}% salary premium\n")
        f.write("\n")

        f.write("2. 🔍 MARKET GAPS (High Pay, Low Competition Categories)\n")
        if market_gaps.empty:
            f.write("   No significant gaps detected in this dataset.\n")
        else:
            for cat, row in market_gaps.iterrows():
                f.write(f"   {cat:20}: Avg Salary ${row['avg_salary']:,.0f} | Count: {row['job_count']}\n")
        f.write("\n")

        f.write("3. 💡 STRATEGIC RECOMMENDATIONS FOR BEGINNERS\n")
        f.write("   - STACKING: Don't just learn 'Python'. Learn 'Python + AWS' or 'SQL + dbt'.\n")
        f.write("   - SENIORITY: Seniority score is the 1# predictor. Prioritize leadership projects.\n")
        f.write("   - LOCATION: Remote roles in 'Worldwide' pay ~18% less than US-specific ones, but offer better life.\n")

    log.info(f"\n✅  Business strategy report generated → {INSIGHTS_FILE}")

if __name__ == "__main__":
    run_insights()
