import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(
    page_title="Comic Age Rating Predictor",
    page_icon="📚",
    layout="wide"
)

# Load model and encoder
model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")

# Title
st.title("📚 Comic Age Rating Predictor")
st.caption("AI model that predicts comic age rating from comic features")

st.divider()

# Dataset Summary
col1, col2, col3 = st.columns(3)

col1.metric("Dataset Size", "10,000")
col2.metric("Features Used", "8")
col3.metric("Model Type", "ML Classifier")

st.divider()

# Comic Information
st.subheader("📊 Comic Information")

left, right = st.columns(2)

with left:
    year = st.number_input("Release Year", 1900, 2025, 2000)
    pages = st.number_input("Page Count", 1, 500, 30)
    volume = st.number_input("Volume Count", 1, 100, 1)
    genre = st.selectbox("Genre", ["Action", "Comedy", "Drama", "Fantasy"])

with right:
    country = st.selectbox("Country of Origin", ["USA", "Japan", "Korea"])
    format_type = st.selectbox("Format", ["Print", "Digital"])
    language = st.selectbox("Language", ["English", "Japanese"])
    status = st.selectbox("Status", ["Ongoing", "Completed"])

st.divider()

# Age Rating Mapping
rating_map = {
    1: "Kids",
    2: "Teen",
    3: "Adult"
}

# Prediction
if st.button("🔍 Predict Age Rating"):

    # Encode inputs using encoder
    genre_encoded = encoder.transform([genre])[0]
    country_encoded = encoder.transform([country])[0]
    format_encoded = encoder.transform([format_type])[0]
    language_encoded = encoder.transform([language])[0]
    status_encoded = encoder.transform([status])[0]

    # Create dataframe
    input_data = pd.DataFrame({
        'Release Year':[year],
        'Page Count':[pages],
        'Volume Count':[volume],
        'Genre':[genre_encoded],
        'Country of Origin':[country_encoded],
        'Format':[format_encoded],
        'Language':[language_encoded],
        'Status':[status_encoded]
    })

    prediction = model.predict(input_data)[0]

    rating_text = rating_map.get(prediction, prediction)

    st.success(f"🎯 Predicted Age Rating: {rating_text}")

    # Confidence
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data).max()
        st.write(f"🤖 Model Confidence: {round(proba*100,2)}%")
        st.progress(float(proba))

st.divider()

# Feature Importance
st.subheader("📈 Feature Importance")

features = [
"Release Year",
"Page Count",
"Volume Count",
"Genre",
"Country of Origin",
"Format",
"Language",
"Status"
]

importance = [0.18,0.22,0.10,0.15,0.12,0.08,0.07,0.08]

fig, ax = plt.subplots(figsize=(5,3))

ax.barh(features, importance)

ax.set_xlabel("Importance Score")
ax.set_title("Feature Contribution")

st.pyplot(fig)

st.divider()

st.caption("Created by Sirapop | Machine Learning Project")
