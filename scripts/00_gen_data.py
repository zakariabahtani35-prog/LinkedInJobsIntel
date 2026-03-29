import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path

def generate_synthetic_jobs(num_rows: int = 1500):
    """Generates a high-quality synthetic dataset for LinkedIn Jobs analysis."""
    companies = ["TechCorp", "DataFlow", "AI-Systems", "CloudNative", "ByteMasters", "SoftGrid", "FutureAI"]
    titles = [
        "Data Scientist", "Senior Data Engineer", "Machine Learning Specialist",
        "Junior Data Analyst", "DevOps Engineer", "Cloud Architect", "Full Stack Developer"
    ]
    locations = ["US", "Canada", "Germany", "UK", "France", "Netherlands", "Worldwide"]
    categories = ["Data", "Software", "Cloud", "Business", "Design"]
    skills_pool = ["Python", "SQL", "AWS", "Machine Learning", "Kubernetes", "Power BI", "Tableau", "Rust", "Go", "Docker", "Spark"]

    data = []
    for i in range(num_rows):
        title = random.choice(titles)
        seniority = "Senior" if "Senior" in title or random.random() > 0.7 else "Junior" if "Junior" in title else "Mid"
        
        # Base salary logic with variability
        base_sal = 110000 if "Senior" in seniority else 65000 if "Junior" in seniority else 85000
        loc_mult = 1.3 if random.choice(locations) == "US" else 1.0
        
        # Skill-based salary premium
        skills = random.sample(skills_pool, k=random.randint(2, 5))
        skill_premium = 0
        if "Machine Learning" in skills: skill_premium += 15000
        if "AWS" in skills and "Kubernetes" in skills: skill_premium += 20000
        if "Python" in skills and "SQL" in skills: skill_premium += 5000
        
        salary_val = (base_sal + skill_premium + random.randint(-5000, 15000)) * loc_mult
        
        data.append({
            "id": 1000 + i,
            "title": title,
            "company": random.choice(companies),
            "category": random.choice(categories),
            "tags": ", ".join(skills),
            "location": random.choice(locations),
            "salary": f"${int(salary_val):,}",
            "published_at": (datetime.now() - timedelta(days=random.randint(0, 60))).isoformat()
        })

    df = pd.DataFrame(data)
    # Add some messy data for demonstrating Cleaner resilience
    df.loc[0, "salary"] = "Competitive"
    df.loc[1, "salary"] = "$45/hr"
    df.loc[2, "id"] = 1000  # Duplicate
    
    out_path = Path("data/raw/jobs_raw.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Generated {num_rows} synthetic job records at {out_path}")

if __name__ == "__main__":
    generate_synthetic_jobs()
