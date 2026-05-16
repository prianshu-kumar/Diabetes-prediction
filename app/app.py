import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "model.pkl"

st.title("Diabetes Prediction")
st.markdown(
    "Use the saved model to predict whether a person is likely to have diabetes based on clinical features."
)

st.sidebar.header("Patient data")

pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=300, value=120)
blood_pressure = st.sidebar.number_input("BloodPressure", min_value=0, max_value=200, value=70)
skin_thickness = st.sidebar.number_input("SkinThickness", min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=1000, value=79)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, format="%.1f")
diabetes_pedigree_function = st.sidebar.number_input(
    "DiabetesPedigreeFunction", min_value=0.0, max_value=5.0, value=0.5, format="%.3f"
)
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)

input_data = np.array([
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree_function,
    age,
]).reshape(1, -1)

st.subheader("Input values")
input_df = pd.DataFrame(
    input_data,
    columns=[
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ],
)
st.write(input_df)

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(f"Model file not found at {MODEL_PATH}")
    st.stop()

if st.button("Predict"):
    prediction = model.predict(input_data)
    label = "Diabetes" if prediction[0] == 1 else "No diabetes"
    st.success(f"Prediction: {label}")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0]
        st.write(
            {
                "No diabetes probability": f"{proba[0]:.2f}",
                "Diabetes probability": f"{proba[1]:.2f}",
            }
        )

