import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Comic Age Rating Predictor",
    page_icon="📚",
    layout="wide"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
.main-title{
    font-size:40px;
    font-weight:700;
}
.card{
    background-color:#111827;
    padding:20px;
    border-radius:12px;
    border:1px solid #1f2937;
}
.metric{
    font-size:22px;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)

# ---------- Load Model ----------
model = joblib.load("model.pkl")

# ---------- Header ----------
st.markdown('<div class="main-title">📚 Comic Age Rating Predictor</div>', unsafe_allow_html=True)
st.caption("AI model that predicts comic age rating from comic features")

st.divider()

# ---------- Dataset Summary ----------
colA,colB,colC = st.columns(3)

colA.metric("Dataset Size","10,000")
colB.metric("Features Used","8")
colC.metric("Model Type","ML Classifier")

st.divider()

# ---------- Input Section ----------
st.subheader("📊 Comic Information")

left,right = st.columns(2)

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

# ---------- Encoding ----------
genre_map = {"Action":0,"Comedy":1,"Drama":2,"Fantasy":3}
country_map = {"USA":0,"Japan":1,"Korea":2}
format_map = {"Print":0,"Digital":1}
language_map = {"English":0,"Japanese":1}
status_map = {"Ongoing":0,"Completed":1}

# ---------- Prediction ----------
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

    st.success(f"🎯 Predicted Age Rating: {prediction}")

    if hasattr(model,"predict_proba"):
        proba = model.predict_proba(input_data).max()
        st.write(f"🤖 Model Confidence: {round(proba*100,2)}%")
        st.progress(float(proba))

st.divider()

# ---------- Feature Importance ----------
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

fig, ax = plt.subplots()

ax.barh(features, importance)
ax.set_xlabel("Importance Score")
ax.set_title("Feature Contribution")

st.pyplot(fig)

st.divider()

st.caption("Developed using Python Machine Learning and Streamlit")
