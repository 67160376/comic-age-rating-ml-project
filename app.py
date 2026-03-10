import streamlit as st
import joblib
import numpy as np

# Page config
st.set_page_config(
    page_title="Comic Age Rating Predictor",
    page_icon="📚",
    layout="centered"
)

# Title
st.title("📚 Comic Age Rating Predictor")

st.markdown(
"""
This app predicts **Comic Book Age Rating** using a Machine Learning model.

---
"""
)

# Input section
st.subheader("📥 Input Features")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input(
        "Release Year",
        min_value=1900,
        max_value=2025,
        value=2000
    )

with col2:
    pages = st.number_input(
        "Page Count",
        min_value=1,
        max_value=500,
        value=30
    )

st.write("---")

# Load model
model = joblib.load("model.pkl")

# Predict
if st.button("🔮 Predict"):

    input_data = np.array([[year, pages]])
    prediction = model.predict(input_data)[0]

    proba = model.predict_proba(input_data).max()

    st.success(f"🎯 Predicted Age Rating: **{prediction}**")

    st.progress(float(proba))

    st.info(f"Model confidence: {round(proba*100,2)}%")

st.write("---")

st.caption(
"⚠️ Disclaimer: This prediction is based on a machine learning model and may not be 100% accurate."
)
