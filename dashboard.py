"""
============================================================
SR — LINKEDIN JOBS INTELLIGENCE DASHBOARD
============================================================
Production-ready Streamlit interface for salary prediction
and job market analytics.
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from pathlib import Path

# Page Config (Premium Aesthetic)
st.set_page_config(
    page_title="LinkedIn Jobs Intelligence",
    page_icon="📊",
    layout="wide"
)

# ── 1. Load Data and Model Artifacts ─────────────────────
@st.cache_resource
def load_assets():
    """Caches large static assets for production performance."""
    # Data for EDA
    data_path = Path("data/processed/enterprise_jobs.parquet")
    df = pd.read_parquet(data_path) if data_path.exists() else None
    
    # Model
    model_path = Path("models/xgb_salary_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    # Transformers Metadata
    meta_path = Path("models/transformers/mlb_classes.pkl")
    with open(meta_path, "rb") as f:
        skill_cols = pickle.load(f)
        
    return df, model, skill_cols

df, model, skill_cols = load_assets()

# ── 2. Sidebar Application ────────────────────────────────
st.sidebar.image("https://img.icons8.com/isometric/100/business-analytics.png", width=80)
st.sidebar.title("Configuration")
st.sidebar.info("Adjust profile to see real-time market valuation.")

role_options = ["Analyst", "Cloud/DevOps", "Data Engineer", "Data Scientist", "Software"]
location_options = ["Worldwide", "US", "UK", "Canada", "Germany", "France", "Netherlands"]

user_role = st.sidebar.selectbox("Your Role Broad Category", role_options)
user_location = st.sidebar.selectbox("Target Location", location_options)
user_seniority = st.sidebar.select_slider("Seniority Level", options=["Junior", "Mid", "Senior"], value="Mid")

# Skill selection based on trained model classes
available_skills = [s.replace("skill_", "").replace("_", " ").title() for s in skill_cols]
user_skills = st.sidebar.multiselect("Select Your Skills", available_skills, default=available_skills[:3])

# ── 3. Main Dashboard ────────────────────────────────────
st.title("🚀 LinkedIn Jobs Intelligence System")
tab1, tab2 = st.tabs(["💰 Salary Predictor", "📈 Market Trends"])

with tab1:
    st.subheader("Personalized Salary Estimation")
    col1, col2 = st.columns([2, 1])
    
    # ML Prediction Logic
    raw_skills = [f"skill_{s.lower().replace(' ', '_')}" for s in user_skills]
    seniority_map = {"Junior": 0, "Mid": 1, "Senior": 2}
    
    # Constructing the Feature Vector (32 dims)
    # We must match the column order from Step 5 perfectly
    training_cols = [
        'seniority_numeric', 'skill_count', 'post_month', 'post_weekday', 'job_age_days',
        'skill_rust', 'skill_power_bi', 'skill_aws', 'skill_go', 'skill_sql', 
        'skill_kubernetes', 'skill_python', 'skill_spark', 'skill_tableau', 
        'skill_machine_learning', 'skill_docker', 'role_Analyst', 'role_Cloud/DevOps', 
        'role_Data Engineer', 'role_Data Scientist', 'role_Software', 'loc_Canada', 
        'loc_France', 'loc_Germany', 'loc_Netherlands', 'loc_UK', 'loc_US', 'loc_Worldwide'
    ]
    
    input_data = pd.DataFrame(0, index=[0], columns=training_cols)
    input_data.at[0, "seniority_numeric"] = seniority_map[user_seniority]
    input_data.at[0, "skill_count"] = len(user_skills)
    input_data.at[0, "post_month"] = 3 # Current Month
    input_data.at[0, "post_weekday"] = 4 # Friday
    input_data.at[0, "job_age_days"] = 5
    
    # Setting flags
    for rs in raw_skills:
        if rs in input_data.columns:
            input_data.at[0, rs] = 1
            
    input_data.at[0, f"role_{user_role}"] = 1
    input_data.at[0, f"loc_{user_location}"] = 1
    
    # Final Prediction
    with col1:
        log_pred = model.predict(input_data)[0]
        final_usd = np.expm1(log_pred)
        
        st.metric(label="Estimated Annual Salary (USD)", value=f"${final_usd:,.0f}")
        st.write("---")
        st.markdown(f"""
        **Insight**: Your profile with **{len(user_skills)} skills** as a **{user_seniority} {user_role}** in **{user_location}** 
        places you in the target bracket for enterprise-level compensation.
        """)
        
    with col2:
        st.write("**Top Predictors for You**")
        # Show mini Importance
        imp_df = pd.read_csv("reports/models/feature_importance.csv").head(5)
        st.dataframe(imp_df, hide_index=True)

with tab2:
    st.subheader("Global Job Market Analytics")
    if df is not None:
        c1, c2 = st.columns(2)
        with c1:
            # Distribution from STEP 3 logic
            fig1 = px.histogram(df, x="salary_numeric", title="Global Salary Distribution", template="plotly_dark", color_discrete_sequence=["#00CC96"])
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            # Multi-Skill analysis
            avg_loc = df.groupby("location")["salary_numeric"].mean().reset_index()
            fig2 = px.bar(avg_loc, x="location", y="salary_numeric", title="Average Salary by Region", template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)
            
    # Business Insights injection
    st.info("💡 Strategic Insights (Derived from Pipeline)")
    if Path("reports/eda/business_insights.txt").exists():
        with open("reports/eda/business_insights.txt", "r") as f:
            st.text(f.read())

st.write("---")
st.caption("SR Intelligence Product | Enterprise Data Pipeline v1.0")
