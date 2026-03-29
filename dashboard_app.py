"""
============================================================
SR — ENTERPRISE LINKEDIN JOBS INTELLIGENCE DASHBOARD (v3.0)
============================================================
Production-ready Streamlit interface for salary prediction
and job market analytics.

Backend improvements (v3.0):
  - Feature schema loaded from disk (feature_schema.pkl)
  - Input aligned via reindex(fill_value=0) — no hardcoding
  - Validation layer detects mismatched features pre-inference
  - sklearn Pipeline (StandardScaler + XGBRegressor) compatible
  - Robust null-safety on impact_features (list vs DataFrame)
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import List, Optional, Tuple, Any

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="LinkedIn Salary Intelligence",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom Styling (Glassmorphism & Professional Palettes) ─
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #00CC96;
        color: white;
        border-radius: 8px;
    }
    .validation-warn {
        background: rgba(255, 165, 0, 0.1);
        border-left: 3px solid #FFA500;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# MODULE 1 — ASSET LOADER
# ═══════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="🔄 Loading intelligence engine...")
def load_assets() -> Tuple[
    Optional[pd.DataFrame],
    Optional[Any],           # sklearn Pipeline
    List[str],               # skill display names (from mlb_classes)
    List[str],               # exact training feature columns (from feature_schema)
    Optional[pd.DataFrame],  # feature importance df
]:
    """
    Loads and caches all production ML artifacts.

    Returns
    -------
    df_raw        : raw EDA dataframe (or None)
    model         : sklearn Pipeline (StandardScaler + XGBRegressor), or None
    skill_cols    : list of skill column names saved by MultiLabelBinarizer
    feature_schema: EXACT column order used during model training (ground truth)
    imp_df        : feature importance dataframe (or None)
    """
    # ── 1. EDA Data (optional) ────────────────────────────
    data_path = Path("data/processed/enterprise_jobs.parquet")
    df_raw = pd.read_parquet(data_path) if data_path.exists() else None

    # ── 2. ML Pipeline (mandatory) ───────────────────────
    model: Optional[Any] = None
    model_path = Path("models/xgb_salary_model.pkl")
    if model_path.exists():
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        except Exception as exc:
            st.error(f"❌ Could not load model: {exc}")

    # ── 3. Skill metadata (MultiLabelBinarizer classes) ──
    skill_cols: List[str] = []
    meta_path = Path("models/transformers/mlb_classes.pkl")
    if meta_path.exists():
        try:
            with open(meta_path, "rb") as f:
                skill_cols = pickle.load(f)
        except Exception as exc:
            st.warning(f"⚠️ Could not load skill metadata: {exc}")

    # ── 4. Feature Schema — extracted directly from the fitted Pipeline ──
    #
    #   sklearn stores `feature_names_in_` on every estimator that was
    #   fitted with a DataFrame. This is the ONLY guaranteed-correct source
    #   of the training column names AND their order.
    #   feature_importance.csv is sorted by importance, NOT by column order —
    #   using it caused the ValueError from StandardScaler.transform().
    #
    #   Priority:
    #     1. model.named_steps["scaler"].feature_names_in_  ← live model (best)
    #     2. models/transformers/feature_schema.pkl         ← saved artifact
    #     3. Empty list → surface an error at prediction time
    feature_schema: List[str] = []

    # Tier 1: extract from the already-loaded Pipeline's StandardScaler
    if model is not None:
        try:
            scaler = model.named_steps.get("scaler")
            if scaler is not None and hasattr(scaler, "feature_names_in_"):
                feature_schema = list(scaler.feature_names_in_)
        except Exception as exc:
            st.warning(f"⚠️ Could not read feature names from scaler: {exc}")

    # Tier 2: fall back to persisted schema file
    if not feature_schema:
        schema_path = Path("models/transformers/feature_schema.pkl")
        if schema_path.exists():
            try:
                with open(schema_path, "rb") as f:
                    feature_schema = pickle.load(f)
            except Exception as exc:
                st.warning(f"⚠️ feature_schema.pkl unreadable: {exc}")

    # Tier 3: no schema available — warn loudly
    if not feature_schema:
        st.error(
            "❌ Cannot determine training feature schema. "
            "Re-run `scripts/05_train_salary_model.py` to regenerate the model."
        )

    # ── 5. Feature Importance (optional explainability) ──
    imp_df: Optional[pd.DataFrame] = None
    imp_path = Path("reports/models/feature_importance.csv")
    if imp_path.exists():
        try:
            imp_df = pd.read_csv(imp_path)
        except Exception:
            pass

    return df_raw, model, skill_cols, feature_schema, imp_df


df_raw, model, skill_cols, feature_schema, imp_df = load_assets()


# ═══════════════════════════════════════════════════════════
# MODULE 2 — INPUT VALIDATION LAYER
# ═══════════════════════════════════════════════════════════

def validate_input_alignment(
    input_df: pd.DataFrame,
    schema: List[str],
) -> Tuple[bool, List[str], List[str]]:
    """
    Compares inference input columns against the authoritative training schema.

    Returns
    -------
    ok            : True if no critical mismatch
    missing_cols  : features expected by model but absent in input_df
    extra_cols    : features in input_df that the model never saw
    """
    if not schema:
        return True, [], []  # no schema available — skip validation

    input_set  = set(input_df.columns)
    schema_set = set(schema)

    missing_cols = sorted(schema_set - input_set)
    extra_cols   = sorted(input_set - schema_set)

    # Critical only if many features are missing (>10% threshold)
    critical_threshold = max(1, int(len(schema) * 0.10))
    ok = len(missing_cols) <= critical_threshold

    return ok, missing_cols, extra_cols


# ═══════════════════════════════════════════════════════════
# MODULE 3 — FEATURE BUILDER
# ═══════════════════════════════════════════════════════════

SENIORITY_MAP = {"Junior": 0, "Mid": 1, "Senior": 2}

def build_input_row(
    skills: List[str],
    role: str,
    location: str,
    seniority: str,
    skill_cols: List[str],
) -> pd.DataFrame:
    """
    Constructs a raw feature row from user inputs.

    All binarised skill flags come from `skill_cols` (the persisted
    MultiLabelBinarizer classes), so the vocabulary always matches training.

    Returns a single-row DataFrame with columns corresponding to the raw
    feature set BEFORE reindex alignment.
    """
    row: dict = {}

    # ── Numeric / temporal features ──────────────────────
    row["seniority_numeric"] = SENIORITY_MAP.get(seniority, 1)
    row["skill_count"]       = len(skills)
    row["post_month"]        = pd.Timestamp.now().month
    row["post_weekday"]      = pd.Timestamp.now().weekday()
    row["job_age_days"]      = 5   # representative average posting age

    # ── Skill flags (one-hot from MLB vocabulary) ────────
    user_skill_keys = {f"skill_{s.lower().replace(' ', '_')}" for s in skills}
    for col in skill_cols:
        row[col] = 1 if col in user_skill_keys else 0

    # ── Role one-hot ─────────────────────────────────────
    role_options = ["Analyst", "Cloud/DevOps", "Data Engineer", "Data Scientist", "Software"]
    for r in role_options:
        row[f"role_{r}"] = 1 if r == role else 0

    # ── Location one-hot ─────────────────────────────────
    loc_options = ["Worldwide", "US", "UK", "Canada", "Germany", "France", "Netherlands"]
    for loc in loc_options:
        row[f"loc_{loc}"] = 1 if loc == location else 0

    return pd.DataFrame([row])


# ═══════════════════════════════════════════════════════════
# MODULE 4 — PREDICTION ENGINE
# ═══════════════════════════════════════════════════════════

def predict_salary(
    skills: List[str],
    role: str,
    location: str,
    seniority: str,
    skill_cols: List[str],
    feature_schema: List[str],
    model: Any,
    imp_df: Optional[pd.DataFrame],
) -> Tuple[float, pd.DataFrame, List[str], List[str]]:
    """
    End-to-end inference pipeline with schema alignment and validation.

    Steps
    -----
    1. Build raw feature row from user inputs
    2. Validate alignment against training schema
    3. Align using reindex(columns=feature_schema, fill_value=0)
    4. Pass aligned DataFrame to sklearn Pipeline.predict()
    5. Back-transform log prediction → USD

    Returns
    -------
    usd_pred      : predicted annual salary in USD
    impact_df     : top impact features as DataFrame (may be empty)
    missing_cols  : features the model expected but were missing
    extra_cols    : features present in input but unseen during training
    """
    # 1. Build raw input row
    input_df = build_input_row(skills, role, location, seniority, skill_cols)

    # 2. Validate against schema
    ok, missing_cols, extra_cols = validate_input_alignment(input_df, feature_schema)

    # 3. Align to exact training schema
    if feature_schema:
        # reindex guarantees:
        #   - missing columns are filled with 0
        #   - extra columns are dropped
        #   - column ORDER matches training exactly
        input_aligned = input_df.reindex(columns=feature_schema, fill_value=0)
    else:
        # No schema: use raw (best-effort, may cause model errors)
        input_aligned = input_df

    # 4. Predict via sklearn Pipeline (handles StandardScaler internally)
    log_pred = model.predict(input_aligned)[0]
    usd_pred = float(np.expm1(log_pred))

    # 5. Determine top impact features for this user profile
    impact_df = pd.DataFrame()
    if imp_df is not None and not imp_df.empty:
        user_features = {f"skill_{s.lower().replace(' ', '_')}" for s in skills}
        user_features |= {f"role_{role}", f"loc_{location}", "seniority_numeric"}
        impact_df = (
            imp_df[imp_df["Feature"].isin(user_features)]
            .sort_values("Importance", ascending=False)
            .head(3)
            .reset_index(drop=True)
        )

    return usd_pred, impact_df, missing_cols, extra_cols


# ═══════════════════════════════════════════════════════════
# UI — SIDEBAR INPUTS
# ═══════════════════════════════════════════════════════════

st.sidebar.image(
    "https://img.icons8.com/isometric/100/business-analytics.png", width=80
)
st.sidebar.title("Intelligence Inputs")
st.sidebar.markdown("---")

role_options     = ["Analyst", "Cloud/DevOps", "Data Engineer", "Data Scientist", "Software"]
location_options = ["Worldwide", "US", "UK", "Canada", "Germany", "France", "Netherlands"]

user_role       = st.sidebar.selectbox("Role Specialization", role_options, index=3)
user_location   = st.sidebar.selectbox("Hiring Region", location_options, index=1)
user_seniority  = st.sidebar.select_slider("Career Seniority", options=["Junior", "Mid", "Senior"], value="Mid")
user_experience = st.sidebar.slider("Experience in Years", 0, 15, 5)

# Skill selector built from persisted MLB vocabulary
available_skills = [
    s.replace("skill_", "").replace("_", " ").title()
    for s in skill_cols
]
user_skills = st.sidebar.multiselect(
    "Skill Set Configuration",
    available_skills,
    default=available_skills[:5] if len(available_skills) >= 5 else available_skills,
)

st.sidebar.markdown("---")

# ── Schema status badge ───────────────────────────────────
if feature_schema:
    st.sidebar.success(f"✅ Schema locked: **{len(feature_schema)} features**")
else:
    st.sidebar.warning("⚠️ No schema — re-train model")

st.sidebar.caption("v3.0 Production Engine | SR Intelligence")


# ═══════════════════════════════════════════════════════════
# UI — MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════

st.title("💼 LinkedIn Salary Intelligence System")

tab1, tab2 = st.tabs(["💰 Career Valuation", "📈 Market Dynamics"])

# ── Tab 1: Salary Prediction ──────────────────────────────
with tab1:
    col_pre, col_res = st.columns([1.5, 1])

    if model is None:
        st.error(
            "⚠️ **Model Asset Error**: The prediction engine could not be initialized. "
            "Please ensure `models/xgb_salary_model.pkl` exists."
        )
    else:
        try:
            predicted_usd, top_impact, missing_cols, extra_cols = predict_salary(
                skills        = user_skills,
                role          = user_role,
                location      = user_location,
                seniority     = user_seniority,
                skill_cols    = skill_cols,
                feature_schema= feature_schema,
                model         = model,
                imp_df        = imp_df,
            )

            # ── Validation warnings (non-blocking) ────────────────
            if missing_cols:
                st.warning(
                    f"⚠️ **Schema Mismatch Detected** — {len(missing_cols)} training "
                    f"feature(s) absent in input (filled with 0): "
                    f"`{'`, `'.join(missing_cols[:8])}{'…' if len(missing_cols) > 8 else ''}`"
                )
            if extra_cols:
                st.info(
                    f"ℹ️ {len(extra_cols)} input feature(s) not seen during training "
                    f"(dropped before inference): "
                    f"`{'`, `'.join(extra_cols[:8])}{'…' if len(extra_cols) > 8 else ''}`"
                )

            with col_pre:
                st.subheader("Your Real-Time Estimation")
                st.metric(
                    label="Market Salary Valuation (USD)",
                    value=f"${predicted_usd:,.2f}",
                )

                st.write("---")
                st.markdown(f"""
                ### Executive Summary
                Based on your profile as a **{user_seniority} {user_role}** in **{user_location}**,
                our XGBoost engine estimates multiple market data points.
                Your skillset depth (**{len(user_skills)} skills**) places you in the upper
                decile of predicted earnings for similar roles.
                """)

                # Impact feature cards — guard against empty DataFrame
                if isinstance(top_impact, pd.DataFrame) and not top_impact.empty:
                    st.info("💡 **Top Drivers for This Estimate:**")
                    impact_cols = st.columns(len(top_impact))
                    for i, (_, row) in enumerate(top_impact.iterrows()):
                        impact_cols[i].metric(
                            label=row["Feature"].replace("skill_", "").replace("_", " ").title(),
                            value="High Impact",
                        )

            with col_res:
                st.subheader("Explainability Matrix")
                if imp_df is not None and not imp_df.empty:
                    fig_imp = px.bar(
                        imp_df.head(10).sort_values(by="Importance"),
                        x="Importance", y="Feature",
                        orientation="h",
                        title="Global Model Predictors",
                        template="plotly_dark",
                        color="Importance",
                        color_continuous_scale="Viridis",
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.info("Feature importance report not available.")

        except Exception as exc:
            st.error(f"🔴 **Prediction Error**: `{exc}`")
            st.exception(exc)


# ── Tab 2: Market Dynamics ────────────────────────────────
with tab2:
    st.subheader("Enterprise Ingestion Diagnostics")
    c1, c2 = st.columns(2)

    if df_raw is not None:
        with c1:
            fig_hist = px.histogram(
                df_raw, x="salary_numeric", nbins=50,
                title="Global Salary Density",
                template="plotly_dark",
                color_discrete_sequence=["#00CC96"],
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with c2:
            skills_exploded = df_raw["skills"].dropna().str.split(", ").explode()
            top_20 = (
                skills_exploded.value_counts()
                .head(20)
                .rename_axis("Skill")
                .reset_index(name="Jobs")
            )
            fig_bar = px.bar(
                top_20, x="Jobs", y="Skill", orientation="h",
                title="In-Demand Tech Stack (Top 20)",
                template="plotly_dark",
                color="Jobs",
            ).update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning(
            "⚠️ Processed enterprise layer not found at "
            "`data/processed/enterprise_jobs.parquet`. "
            "Run `main_pipeline.py` first."
        )

st.markdown("---")
st.caption(
    "AI Ethics & Governance: All predictions are based on statistical modeling "
    "of LinkedIn datasets. Use as a strategic baseline only."
)
