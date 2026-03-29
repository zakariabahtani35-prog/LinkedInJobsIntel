"""
============================================================
SR — ENTERPRISE JOB INTELLIGENCE PRO (v4.0)
============================================================
The premium, production-grade interface for high-end
job market analytics and career strategy.
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import shap
from pathlib import Path
from typing import List, Optional, Tuple, Any

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="SR Pro Intelligence Dashboard",
    page_icon="💎",
    layout="wide"
)

# ── Modern UI / UX Styling ────────────────────────────────
st.markdown("""
<style>
    /* Glassmorphism / Dark Mode Premium */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(10, 15, 30) 0%, rgb(5, 5, 10) 90%);
        color: #E0E0E0;
    }
    
    .stHeader {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 25px;
    }

    .metric-card {
        background: rgba(0, 204, 150, 0.05);
        border: 1px solid rgba(0, 204, 150, 0.2);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        background: rgba(0, 204, 150, 0.08);
    }
    
    .skill-tag {
        display: inline-block;
        padding: 5px 12px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        margin: 4px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        font-size: 0.85rem;
    }

    .stButton>button {
        background: linear-gradient(90deg, #00CC96, #00A3CC);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        opacity: 0.9;
        box-shadow: 0 4px 15px rgba(0, 204, 150, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ── Load Core Assets (from previous work) ─────────────────
@st.cache_resource
def load_pro_assets():
    model_path = Path("models/xgb_salary_model.pkl")
    data_path = Path("data/processed/enterprise_jobs.parquet")
    meta_path = Path("models/transformers/mlb_classes.pkl")
    
    # Placeholder for a real model loader (reuse load_assets from v3)
    # We load them from disk if possible
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(meta_path, "rb") as f:
        skill_cols = pickle.load(f)
    df_raw = pd.read_parquet(data_path) if data_path.exists() else None
    
    # Retrieve feature schema from scaler (as per b51a8aa5 discussion)
    feature_schema = list(model.named_steps["scaler"].feature_names_in_)
    
    return df_raw, model, skill_cols, feature_schema

df_raw, model, skill_cols, feature_schema = load_pro_assets()

# ── 🏗️ Utility: Profile Builder ───────────────────────────
def build_profile_vector(skills, role, loc, seniority, skill_cols, schema):
    row = {col: 0 for col in schema}
    row["seniority_numeric"] = {"Junior": 0, "Mid": 1, "Senior": 2}.get(seniority, 1)
    row["skill_count"] = len(skills)
    row["post_month"] = pd.Timestamp.now().month
    row["post_weekday"] = pd.Timestamp.now().weekday()
    row["job_age_days"] = 5
    
    for s in skills:
        key = f"skill_{s.lower().replace(' ', '_')}"
        if key in row: row[key] = 1
        
    row[f"role_{role}"] = 1
    row[f"loc_{loc}"] = 1
    
    return pd.DataFrame([row]).reindex(columns=schema, fill_value=0)

# ── Main Layout ───────────────────────────────────────────
with st.container():
    st.markdown("""
        <div class='stHeader'>
            <h1 style='margin:0;'>💎 Career Intelligence Pro</h1>
            <p style='margin:0; opacity: 0.7;'>Strategic Salary Simulation & Career ROI Engine</p>
        </div>
    """, unsafe_allow_html=True)

col_ctrl, col_main = st.columns([1, 2.8], gap="large")

with col_ctrl:
    st.subheader("🛠️ Profile Config")
    user_role = st.selectbox("Current Role", ["Data Scientist", "Data Engineer", "Analyst", "Cloud/DevOps"], index=0)
    user_seniority = st.select_slider("Seniority Level", options=["Junior", "Mid", "Senior"], value="Mid")
    user_location = st.selectbox("Location", ["US", "Europe", "UK", "Canada", "Worldwide"], index=0)
    
    available_skills = [s.replace("skill_", "").replace("_", " ").title() for s in skill_cols]
    user_skills = st.multiselect("Active Skills", available_skills, default=available_skills[:5])
    
    st.markdown("---")
    st.subheader("🚀 Future Skills (Simulator)")
    future_skills = st.multiselect("Add Skills for Simulation", 
                                   [s for s in available_skills if s not in user_skills])

# ── Calculations ──
current_vec = build_profile_vector(user_skills, user_role, user_location, user_seniority, skill_cols, feature_schema)
future_vec  = build_profile_vector(user_skills + future_skills, user_role, user_location, user_seniority, skill_cols, feature_schema)

current_sal = float(np.expm1(model.predict(current_vec)[0]))
future_sal  = float(np.expm1(model.predict(future_vec)[0]))
increase = future_sal - current_sal

with col_main:
    # ── 1. The Metric Panel ──
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"<div class='metric-card'><h3>Current Valuation</h3><h2 style='color:#00CC96'>${current_sal:,.0f}</h2></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-card'><h3>Future Potential</h3><h2>${future_sal:,.0f}</h2></div>", unsafe_allow_html=True)
    with m3:
        color = "#00CC96" if increase > 0 else "#FF4B4B"
        st.markdown(f"<div class='metric-card'><h3>Skill ROI</h3><h2 style='color:{color}'>+${increase:,.0f}</h2></div>", unsafe_allow_html=True)
    
    st.write("")
    
    # ── 2. Strategy Engine ──
    tab1, tab2, tab3 = st.tabs(["📊 ROI Comparison", "🧬 Local Explainability (SHAP)", "📈 Growth Trajectory"])
    
    with tab1:
        st.subheader("Skill Contribution Dynamics")
        # Comparison Chart
        labels = ["Current Profile", "Simulated Profile"]
        salaries = [current_sal, future_sal]
        fig = px.bar(x=labels, y=salaries, color=labels, 
                     color_discrete_map={"Current Profile": "#555", "Simulated Profile": "#00CC96"},
                     title="Immediate Salary Impact Analysis")
        fig.update_layout(showlegend=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        if future_skills:
            st.info(f"💡 Adding **{', '.join(future_skills)}** increases your annual market value by **{(increase/current_sal)*100:.1f}%**.")

    with tab2:
        st.subheader("Inference Breakdown (SHAP)")
        st.markdown("This section shows the *exact* weight each feature contributes to your final salary estimation.")
        
        # SHAP Logic
        xgb_model = model.named_steps["regressor"]
        explainer = shap.TreeExplainer(xgb_model)
        
        # Scaling is handled by the model pipeline, so we use the scaled input
        scaled_input = model.named_steps["scaler"].transform(current_vec)
        shap_values = explainer.shap_values(scaled_input)
        
        # Create custom Bar chart for SHAP
        shap_df = pd.DataFrame({
            "Feature": feature_schema,
            "Impact": shap_values[0]
        }).sort_values(by="Impact", ascending=False).head(10)
        
        fig_shap = px.bar(shap_df, x="Impact", y="Feature", orientation="h",
                          color="Impact", color_continuous_scale="RdYlGn",
                          title="Local Feature Attribution (SHAP)",
                          template="plotly_dark")
        st.plotly_chart(fig_shap, use_container_width=True)

    with tab3:
        st.subheader("Career Progression Forecast")
        # Growth path math
        levels = ["Junior", "Mid", "Senior", "Lead/Staff"]
        mults = [0.7, 1.0, 1.45, 1.9]
        trajectory = [current_sal * m for m in mults]
        
        fig_traj = go.Figure()
        fig_traj.add_trace(go.Scatter(x=levels, y=trajectory, mode='lines+markers', 
                                      line=dict(color='#00CC96', width=4),
                                      marker=dict(size=12)))
        fig_traj.update_layout(title="Predicted Salary Trajectory (USD)", 
                               template="plotly_dark",
                               yaxis_title="Annual Compensation ($)")
        st.plotly_chart(fig_traj, use_container_width=True)

st.markdown("---")
st.markdown("<p style='text-align: center; opacity: 0.5;'>Enterprise Jobs Intelligence Pro · v4.0 Production Release</p>", unsafe_allow_html=True)
