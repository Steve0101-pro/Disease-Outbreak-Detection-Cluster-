🦠 Disease Outbreak Intelligence System

An AI-powered disease outbreak risk assessment platform that combines unsupervised machine learning (Gaussian Mixture Model clustering and KMeans Model), environmental health indicators, and Generative AI (NVIDIA NIM LLM) to provide risk level report, outbreak analysis, and actionable public health recommendations.

📌 Project Overview

Disease outbreaks are often influenced by environmental, demographic, and healthcare-related factors.

This project leverages:

Climate indicators
Air quality data
Population density
Healthcare investment
Seasonal patterns

to identify hidden patterns in disease risk using clustering techniques.

The system classifies an input scenario into a risk cluster and generates an AI-powered interpretation of the results.

🎯 Objectives

The objectives of this project are:

.Identify hidden outbreak risk patterns using clustering.
.Group similar environmental-health conditions into risk categories.
.Provide an interactive dashboard for outbreak assessment.
.Generate AI-driven public health reports.
.Visualize cluster behavior and feature relationships.

🏗️ System Architecture
User Input
│
▼
Feature Engineering
(Month → Sin/Cos)
│
▼
StandardScaler

     │
     ▼

PCA(Dimensional Redundancy)
│
▼
Gaussian Mixture Model/KMeans Model(Based on Model Selected)
│
▼
Cluster Assignment
│
├────────────► Risk Classification
│
▼
NVIDIA LLM Analysis
│
▼
AI Risk Report

📊 Dataset Features

The model uses the following features:

Feature Description
avg_temp_c Average temperature (°C)
precipitation_mm Rainfall amount (mm)
air_quality_index Air pollution level
uv_index UV radiation level
population_density Population concentration
healthcare_budget Healthcare spending
total_cases Recorded disease cases
month_sin Cyclical month encoding
month_cos Cyclical month encoding

🔄 Feature Engineering

Total disease Cases
total_cases = malaria_cases + dengue_cases

Seasonality was captured using cyclical encoding:

month_sin = np.sin(2 _ np.pi _ month / 12)
month_cos = np.cos(2 _ np.pi _ month / 12)
This preserves the cyclical relationship between months.

December → January

🤖 AI Component

The NVIDIA NIM LLM is used to:

🧠 Interpret cluster outputs
📊 Explain risk levels
🏥 Generate public health recommendations
📄 Produce natural language outbreak reports

⚙️ Tech Stack
🐍 Python
🎈 Streamlit
🤖 Scikit-learn
📊 Pandas / NumPy
📉 Matplotlib / Seaborn
💾 Joblib
🧠 NVIDIA NIM LLM

📦 Installation
git clone https://github.com/Steve0101-pro/Disease-Outbreak-Detection-Cluster-.git
cd disease-outbreak-intelligence
pip install -r requirements.txt
streamlit run UI/app.py

🌐 Deployment

Deployed using:

🚀 Streamlit Community Cloud
🔗 GitHub Integration (auto-deploy on push)

📌 Future Improvements

🌍 Real-time climate API integration
🗺️ Geo-mapping outbreak heatmaps
📊 Advanced cluster explainability dashboard
🧠 Fine-tuned medical LLM
💾 Database logging system

👨‍💻 Author

Built as an AI + Machine Learning health intelligence system for outbreak Risk and analysis.
