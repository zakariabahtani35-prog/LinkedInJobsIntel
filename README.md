# 💎 LinkedIn Jobs Intelligence Pro (v4.0)

> **"Transforming raw market data into strategic career leverage."**

![Architecture](https://img.shields.io/badge/Architecture-Medallion-blue)
![Engine](https://img.shields.io/badge/Model-XGBoost-00CC96)
![Insights](https://img.shields.io/badge/Explainability-SHAP-FF4B4B)
![Frontend](https://img.shields.io/badge/Frontend-Streamlit--Pro-white)

## 🎯 The Problem
The modern job market is a "black box." Candidates often:
- Don't know the **actual market value** of their current skills.
- Waste months learning tools that have **low ROI**.
- Lack data-backed evidence for **salary negotiations**.

## 🚀 The Solution
This system is an **End-to-End Artificial Intelligence Product** that scrapes, cleans, models, and explains the LinkedIn job market. It empowers professionals to:
1. **Predict** their market value with 85%+ accuracy.
2. **Simulate** the impact of learning new skills (e.g., "What if I add AWS?").
3. **Analyze** their career trajectory from Junior to Executive levels.

---

## ⚙️ Enterprise Architecture

```text
[ DATA INGESTION ]  ──>  [ DATA REFINERY ]  ──>  [ INTELLIGENCE LAYER ]  ──>  [ PRO DASHBOARD ]
(LinkedIn Scraper)       (Cleaning / FE)         (XGBoost / SHAP)          (Career Simulator)
       │                        │                        │                         │
       └──> Local Parquet       └──> Feature Store       └──> Model Registry       └──> Streamlit UI
```

## 💎 Key Features

- **💰 Smart Salary Simulator**: Real-time "What-If" analysis for skill acquisition.
- **🧬 Deep Explainability (SHAP)**: Local feature attribution showing exactly why you're valued at $X.
- **📈 Career Path Forecaster**: Data-driven salary growth curves based on market benchmarks.
- **📊 Market Dynamics Heatmaps**: High-density visualizations of skill supply vs. demand.

## 🛠️ Tech Stack

- **Data Engineering**: Python, pandas, DuckDB, Parquet.
- **Machine Learning**: XGBoost, scikit-learn (Pipelines), SHAP.
- **Visualization**: Plotly, Streamlit.
- **DevOps**: Docker, GitHub Actions (CI/CD readiness).

## 🚀 Quick Start

1. **Clone & Install**
   ```bash
   git clone https://github.com/yourusername/linkedin-jobs-pro.git
   pip install -r requirements.txt
   ```

2. **Generate Synthetic Data (Optional)**
   ```bash
   python scripts/00_gen_data.py
   ```

3. **Run the Enterprise Pipeline**
   ```bash
   python main_pipeline.py
   ```

4. **Launch the Pro Dashboard**
   ```bash
   streamlit run pro_dashboard.py
   ```

## 📈 Strategic Insights (Sample)
- **The AWS Premium**: Professionals with AWS + Kubernetes see a **~22% median salary increase** over generalist roles.
- **The "Data Gap"**: 70% of high-paying roles require SQL, but only 40% of candidates list it as a primary skill.

---

## 👨‍💻 Author
**[Your Name/Role]**
*Passionate about building production-grade data systems that solve real-world career challenges.*

[LinkedIn Profile](https://linkedin.com) | [Portfolio](https://yourportfolio.com)
