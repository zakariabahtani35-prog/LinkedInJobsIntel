import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import pickle
from pathlib import Path

class IntelligenceEngine:
    """
    Advanced intelligence logic for Salary Simulation, Skill Recommendations,
    and Career Path Analysis.
    """
    def __init__(self, model, feature_schema: List[str], skill_cols: List[str]):
        self.model = model
        self.feature_schema = feature_schema
        self.skill_cols = skill_cols

    def simulate_salary(self, base_profile: pd.DataFrame, added_skills: List[str] = None, 
                        removed_skills: List[str] = None, experience_years: int = None) -> float:
        """
        Simulates salary impact based on profile modifications.
        """
        sim_df = base_profile.copy()
        
        if added_skills:
            for skill in added_skills:
                col = f"skill_{skill.lower().replace(' ', '_')}"
                if col in sim_df.columns:
                    sim_df[col] = 1
        
        if removed_skills:
            for skill in removed_skills:
                col = f"skill_{skill.lower().replace(' ', '_')}"
                if col in sim_df.columns:
                    sim_df[col] = 0

        # Run aligned prediction
        # (Assuming the dashboard alignment logic is reused or integrated here)
        aligned = sim_df.reindex(columns=self.feature_schema, fill_value=0)
        log_pred = self.model.predict(aligned)[0]
        return float(np.expm1(log_pred))

    def recommend_skills(self, current_skills: List[str], role: str, location: str, seniority: str) -> List[Dict]:
        """
        Identifies high-ROI skills for a specific profile.
        """
        # 1. Map current profile to base row
        # (Helper logic to build a standard row)
        recommendations = []
        
        # Get all available skills not in current set
        available_skills_raw = [s.replace("skill_", "").replace("_", " ").title() for s in self.skill_cols]
        target_skills = [s for s in available_skills_raw if s not in current_skills]
        
        # We'll build a dummy profile to test against
        # Simplified: We test top 10 potential skills for impact
        test_profiles = []
        for skill in target_skills[:15]: 
             # Implementation detail: Calculate delta for each skill
             pass

        # For the sake of the initial engine, let's return a structured dummy based on feature importance
        # In a real scenario, we loop and predict.
        return [
            {"skill": "AWS", "impact": 12500, "demand": "High", "reason": "Cloud infrastructure is a top-tier salary driver."},
            {"skill": "Kubernetes", "impact": 9800, "demand": "High", "reason": "Leading tech for scalable deployments."},
            {"skill": "Machine Learning", "impact": 15000, "demand": "Very High", "reason": "AI specialization commands premium rates."}
        ]

    def get_career_trajectory(self, current_salary: float, seniority: str) -> Dict:
        """
        Predicts growth curve from Junior -> Mid -> Senior.
        """
        # Placeholder for growth math
        return {
            "Junior": current_salary * 0.7,
            "Mid": current_salary,
            "Senior": current_salary * 1.45,
            "Lead/Staff": current_salary * 1.8
        }
