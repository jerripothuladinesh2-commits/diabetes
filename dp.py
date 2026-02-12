import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("diabetes_model.pkl")

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

st.title("🩺 Diabetes Risk Prediction System")
st.write("Enter patient details below to check diabetes risk level.")

# -----------------------------
# Sidebar Info
# -----------------------------
st.sidebar.title("About")
st.sidebar.info(
    "This app predicts the probability of diabetes "
    "using a trained Random Forest model."
)

# -----------------------------
# Input Fields
# -----------------------------
age = st.number_input("Age", 1, 100, 30)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
glucose = st.number_input("Glucose Level", 50, 300, 100)
blood_pressure = st.number_input("Blood Pressure", 40, 200, 80)
cholesterol = st.number_input("Cholesterol", 100, 400, 180)
insulin = st.number_input("Insulin", 0, 500, 80)
family_history = st.selectbox("Family History of Diabetes", [0, 1])
stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
sleep_hours = st.slider("Sleep Hours per Day", 1, 12, 7)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Risk"):

    input_data = pd.DataFrame([{
        'age': age,
        'bmi': bmi,
        'glucose': glucose,
        'blood_pressure': blood_pressure,
        'cholesterol': cholesterol,
        'insulin': insulin,
        'family_history': family_history,
        'stress_level': stress_level,
        'sleep_hours': sleep_hours
    }])

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Result")
    st.write("Probability of Diabetes:", round(probability * 100, 2), "%")

    if probability < 0.30:
        st.success("🟢 Low Risk")
    elif probability < 0.70:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")