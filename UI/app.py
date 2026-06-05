import streamlit as st
import os
from dotenv import load_dotenv
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI

load_dotenv()


st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1584036561566-baf8f5f1b144");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .block-container {
        background-color: rgba(0,0,0,0.6);
        padding: 2rem;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ==================================================
# CONFIG
# ==================================================
st.set_page_config(page_title="Outbreak Risk Engine", layout="wide")
st.title("🦠 Disease Outbreak Risk Intelligence Engine")


# ==================================================
# FEATURES
# ==================================================
FEATURES = [
    'avg_temp_c',
    'precipitation_mm',
    'air_quality_index',
    'uv_index',
    'population_density',
    'healthcare_budget',
    'total_cases',
    'month_sin',
    'month_cos'
]

# ==================================================
# LOAD MODELS (MULTI-MODEL SYSTEM)
# ==================================================
@st.cache_resource
def load_models():
    models = {
        "KMeans": joblib.load("Climate_Diseases_kmeans.pkl"),
        "GMM": joblib.load("Climate_Diseases_model.pkl"),
    }

    # optional DBSCAN (if exists)
    try:
        models["DBSCAN"] = joblib.load("dbscan_model.pkl")
    except:
        pass

    return models


@st.cache_resource
def load_scaler():
    return joblib.load("Climate_Diseases_scaler.pkl")

models = load_models()
scaler = load_scaler()

# ==================================================
# SIDEBAR MODEL SELECTOR
# ==================================================
st.sidebar.title("🧠 AI Engine Selector")

selected_model_name = st.sidebar.selectbox(
    "Choose Model",
    list(models.keys())
)

model = models[selected_model_name]

st.sidebar.info(f"Active Model: {selected_model_name}")

# ==================================================
# SAFE INPUT
# ==================================================
def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

def safe_int(x, default=0):
    try:
        return int(x)
    except:
        return default

# ==================================================
# OUTBREAK RISK ENGINE
# ==================================================
def compute_risk_score(temp, rain, aqi, uv, pop, budget, cases):

    score = 0

    score += (temp / 60) * 15
    score += (rain / 600) * 10
    score += (aqi / 500) * 25
    score += (uv / 15) * 10
    score += (pop / 10000) * 15
    score += (1 - min(budget / 10000, 1)) * 10
    score += (cases / 10000) * 25

    return min(score, 100)

def risk_level(score):
    if score < 33:
        return "🟢 Low Risk"
    elif score < 66:
        return "🟡 Medium Risk"
    else:
        return "🔴 High Risk"

# ==================================================
# ML INSIGHT (SAFE FOR ALL MODELS)
# ==================================================
def ml_insight(input_df, scaler, model):

    try:
        X = scaler.transform(input_df[FEATURES])
        return int(model.predict(X)[0])
    except:
        return None

# ==================================================
# UI TABS
# ==================================================
tab1, tab2, tab3 = st.tabs([
    "🎯 Risk Engine",
    "🧠 AI Report",
    "📊 Analytics"
])

# ==================================================
# TAB 1
# ==================================================
with tab1:

    st.subheader("📍 Epidemiological Risk Input System")

    left, right = st.columns([1, 1.2])

    with left:

        st.markdown("### 🧾 Input Parameters")

        temp = safe_float(st.text_input("🌡 Temperature (°C)", "25"))
        rain = safe_float(st.text_input("🌧 Rainfall (mm)", "100"))
        aqi = safe_int(st.text_input("🌫 AQI", "100"))
        uv = safe_int(st.text_input("☀ UV Index", "5"))

        pop = safe_int(st.text_input("🏙 Population Density", "500"))
        budget = safe_int(st.text_input("🏥 Healthcare Budget", "1000"))
        cases = safe_int(st.text_input("🦠 Total Cases", "0"))

        month = st.selectbox("📅 Month", list(range(1, 13)))

        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        run = st.button("🚨 Compute Outbreak Risk", use_container_width=True)

    with right:

        st.markdown("### 📊 Risk Dashboard")

        if run:

            score = compute_risk_score(
                temp, rain, aqi, uv, pop, budget, cases
            )

            level = risk_level(score)

            input_df = pd.DataFrame([{
                "avg_temp_c": temp,
                "precipitation_mm": rain,
                "air_quality_index": aqi,
                "uv_index": uv,
                "population_density": pop,
                "healthcare_budget": budget,
                "total_cases": cases,
                "month_sin": month_sin,
                "month_cos": month_cos
            }])

            cluster = ml_insight(input_df, scaler, model)

            st.metric("🧠 Risk Score", f"{score:.2f} / 100")
            st.metric("⚠ Risk Level", level)

            if cluster is not None:
                st.info(f"ML Insight ({selected_model_name}) Cluster: {cluster}")

            st.progress(score / 100)

            st.session_state.result = {
                "score": score,
                "level": level,
                "input": input_df
            }

        else:
            st.info("Enter values and compute risk")

# ==================================================
# TAB 2
# ==================================================
with tab2:

    st.subheader("🧠 WHO AI Risk Report")

    if "result" not in st.session_state:
        st.warning("Run risk engine first")
    else:

        r = st.session_state.result

        prompt = f"""
You are a WHO epidemiology intelligence system.

Risk Score: {r['score']:.2f}/100
Risk Level: {r['level']}

Temperature: {r['input']['avg_temp_c'][0]}
Rainfall: {r['input']['precipitation_mm'][0]}
AQI: {r['input']['air_quality_index'][0]}
UV: {r['input']['uv_index'][0]}
Population: {r['input']['population_density'][0]}
Healthcare Budget: {r['input']['healthcare_budget'][0]}
Cases: {r['input']['total_cases'][0]}

Provide:
- Executive Summary
- Risk Drivers
- Prevention Plan
"""

        if st.button("Generate AI Report"):

            client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=os.getenv("API_KEY")
            )

            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": "You are a WHO outbreak expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=800,
                stream=True
            )

            output = ""
            box = st.empty()

            for chunk in response:
                if chunk.choices[0].delta.content:
                    output += chunk.choices[0].delta.content
                    box.markdown(output)

# ==================================================
# TAB 3
# ==================================================
with tab3:

    st.subheader("📊 Epidemiological Analytics")

    feature = st.selectbox("Select Feature", FEATURES)

    fig, ax = plt.subplots()
    ax.hist(np.random.normal(0, 1, 100))
    st.pyplot(fig)

    st.markdown("### Model in Use")
    st.write(selected_model_name)