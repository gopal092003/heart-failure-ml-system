import streamlit as st
import requests

API_URL = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Heart Failure Predictor", layout="centered")

st.title("❤️ Heart Failure Prediction")
st.write("Enter patient details to predict risk and see explanations.")

# -------------------------
# Input Form
# -------------------------
with st.form("prediction_form"):
    age = st.slider("Age", 0, 100, 60)
    anaemia = st.selectbox("Anaemia", [0, 1])
    cpk = st.number_input("Creatinine Phosphokinase", value=250)
    diabetes = st.selectbox("Diabetes", [0, 1])
    ef = st.slider("Ejection Fraction", 0, 100, 35)
    hbp = st.selectbox("High Blood Pressure", [0, 1])
    platelets = st.number_input("Platelets", value=250000.0)
    creatinine = st.number_input("Serum Creatinine", value=1.2)
    sodium = st.number_input("Serum Sodium", value=135)
    sex = st.selectbox("Sex", [0, 1])
    smoking = st.selectbox("Smoking", [0, 1])

    submitted = st.form_submit_button("Predict")

# -------------------------
# Prediction
# -------------------------
if submitted:
    payload = {
        "age": age,
        "anaemia": anaemia,
        "creatinine_phosphokinase": cpk,
        "diabetes": diabetes,
        "ejection_fraction": ef,
        "high_blood_pressure": hbp,
        "platelets": platelets,
        "serum_creatinine": creatinine,
        "serum_sodium": sodium,
        "sex": sex,
        "smoking": smoking
    }

    # Prediction
    response = requests.post(f"{API_URL}/predict/", json=payload)

    if response.status_code == 200:
        risk = response.json()["death_risk"]

        st.success(f"Predicted Risk: {risk:.2%}")

        # Explanation
        explain_res = requests.post(f"{API_URL}/explain/", json=payload)

        if explain_res.status_code == 200:
            explanation = explain_res.json()["explanation"]

            st.subheader("🔍 Feature Contributions")

            st.bar_chart(explanation)
    else:
        st.error("Error getting prediction")