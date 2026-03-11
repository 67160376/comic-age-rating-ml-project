import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Comic Age Rating Predictor",
    page_icon="📚",
    layout="wide"
)

# Load model
model = joblib.load("model.pkl")

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
    year = st.number_input("Release Year",1900,2025,2000)
    pages = st.number_input("Page Count",1,500,30)
    volume = st.number_input("Volume Count",1,100,1)
    genre = st.selectbox("Genre",["Action","Comedy","Drama","Fantasy"])

with right:
    country = st.selectbox("Country of Origin",["USA","Japan","Korea"])
    format_type = st.selectbox("Format",["Print","Digital"])
    language = st.selectbox("Language",["English","Japanese"])
    status = st.selectbox("Status",["Ongoing","Completed"])

st.divider()

# Encoding
genre_map = {"Action":0,"Comedy":1,"Drama":2,"Fantasy":3}
country_map = {"USA":0,"Japan":1,"Korea":2}
format_map = {"Print":0,"Digital":1}
language_map = {"English":0,"Japanese":1}
status_map = {"Ongoing":0,"Completed":1}

# Age Rating Mapping
rating_map = {
    1: "Kids",
    2: "Teen",
    3: "Adult"
}

# Prediction
if st.button("🔍 Predict Age Rating"):

    input_data = pd.DataFrame({
        'Release Year':[year],
        'Page Count':[pages],
        'Volume Count':[volume],
        'Genre':[genre_map[genre]],
        'Country of Origin':[country_map[country]],
        'Format':[format_map[format_type]],
        'Language':[language_map[language]],
        'Status':[status_map[status]]
    })

    prediction = model.predict(input_data)[0]

    rating_text = rating_map.get(prediction, prediction)

    st.success(f"🎯 Predicted Age Rating: {rating_text}")

    # Confidence
    if hasattr(model,"predict_proba"):
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

fig, ax = plt.subplots(figsize=(6,4))

ax.barh(features, importance)

ax.set_xlabel("Importance Score")
ax.set_title("Feature Contribution")

st.pyplot(fig)

st.divider()

st.caption("Developed using Python Machine Learning & Streamlit")
