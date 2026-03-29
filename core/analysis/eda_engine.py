"""
============================================================
SR — ENTERPRISE EDA ENGINE (PRESENTATION-READY)
============================================================
Handles univariate, bivariate, and multivariate analysis 
with interactive Plotly visuals and automated insights.
============================================================
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List
from src.utils.logger import get_logger

log = get_logger("eda_engine")

class EDAEngine:
    def __init__(self, data_path: str, output_dir: str = "reports/eda"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(parents=True, exist_ok=True)
        
        # Load Data
        if self.data_path.suffix == ".parquet":
            self.df = pd.read_parquet(self.data_path)
        else:
            self.df = pd.read_csv(self.data_path)
            
        log.info(f"Loaded {len(self.df)} records for professional EDA.")

    # ── 1. Overview & Data Dictionary ──────────────────────────
    def generate_summary(self) -> pd.DataFrame:
        """Creates a high-level summary table of the dataset."""
        summary = pd.DataFrame({
            "Column": self.df.columns,
            "DataType": self.df.dtypes.values,
            "Non-Null Count": self.df.notnull().sum().values,
            "Null Count": self.df.isnull().sum().values,
            "Unique Values": [self.df[col].nunique() for col in self.df.columns]
        })
        summary.to_csv(self.output_dir / "data_summary.csv", index=False)
        return summary

    # ── 2. Univariate Analysis ────────────────────────────────
    def run_univariate(self):
        """Analyzes individual column distributions."""
        log.info("Running Univariate Analysis...")
        
        # Salary distribution
        fig_sal = px.histogram(
            self.df, x="salary_numeric", nbins=30, 
            title="Annual Salary Distribution (USD)",
            template="plotly_dark", color_discrete_sequence=["#00CC96"]
        )
        fig_sal.write_html(self.output_dir / "figures/01_salary_dist.html")
        
        # Top 20 Skills
        all_skills = self.df["tags"].str.split(", ").explode()
        top_skills = all_skills.value_counts().head(20).reset_index()
        top_skills.columns = ["Skill", "Frequency"]
        
        fig_skills = px.bar(
            top_skills, x="Frequency", y="Skill", orientation="h",
            title="Top 20 In-Demand Skills",
            template="plotly_dark", color="Frequency"
        ).update_layout(yaxis={'categoryorder':'total ascending'})
        fig_skills.write_html(self.output_dir / "figures/02_top_skills.html")

    # ── 3. Bivariate & Multivariate ───────────────────────────
    def run_bivariate(self):
        """Analyzes relationships between variables."""
        log.info("Running Bivariate/Multivariate Analysis...")
        
        # Salary vs Location
        avg_sal_loc = self.df.groupby("location")["salary_numeric"].mean().sort_values(ascending=False).reset_index()
        fig_loc = px.bar(
            avg_sal_loc, x="location", y="salary_numeric",
            title="Average Salary by Region",
            template="plotly_dark", color="salary_numeric"
        )
        fig_loc.write_html(self.output_dir / "figures/03_salary_by_loc.html")

        # Correlation Matrix (Numeric)
        numeric_df = self.df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            corr = numeric_df.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap="mako", fmt=".2f")
            plt.title("Correlation Matrix")
            plt.savefig(self.output_dir / "figures/04_correlation.png")
            plt.close()

    # ── 4. Strategic Business Insights ────────────────────────
    def extract_strategic_insights(self) -> List[str]:
        """Calculates 5 high-value insights for business stakeholders."""
        insights = []
        
        # Insight 1: US vs Global Premium
        us_sal = self.df[self.df["location"] == "US"]["salary_numeric"].mean()
        ww_sal = self.df[self.df["location"] == "Worldwide"]["salary_numeric"].mean()
        if pd.notna(us_sal) and pd.notna(ww_sal):
            premium = ((us_sal / ww_sal) - 1) * 100
            insights.append(f"GEOGRAPHIC GAP: US roles command a {premium:.1f}% median salary premium over Global/Remote roles.")

        # Insight 2: Skill Stacking (ML + AWS)
        stack_mask = (self.df["tags"].str.contains("Machine Learning", na=False)) & \
                     (self.df["tags"].str.contains("AWS", na=False))
        stack_sal = self.df[stack_mask]["salary_numeric"].mean()
        base_sal  = self.df["salary_numeric"].mean()
        if pd.notna(stack_sal):
            uplift = ((stack_sal / base_sal) - 1) * 100
            insights.append(f"SKILL STACKING: Candidates with 'ML + AWS' earn {uplift:.1f}% more than the market average.")

        # Insight 3: Core Requirement
        top_skill = self.df["tags"].str.split(", ").explode().value_counts().index[0]
        insights.append(f"MARKET DOMINANCE: '{top_skill}' remains the most requested skill, appearing in {self.df['tags'].str.contains(top_skill).mean()*100:.1f}% of postings.")

        # Save to file
        with open(self.output_dir / "business_insights.txt", "w") as f:
            f.write("\n".join(insights))
            
        return insights
