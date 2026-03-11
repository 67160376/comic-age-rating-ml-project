import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")

st.title("Comic Age Rating Predictor")

year = st.number_input("Release Year",1900,2025,2000)
pages = st.number_input("Page Count",1,500,30)
volume = st.number_input("Volume Count",1,100,1)

genre = st.selectbox("Genre",["Action","Comedy","Drama","Fantasy"])
country = st.selectbox("Country of Origin",["USA","Japan","Korea"])
format_type = st.selectbox("Format",["Print","Digital"])
language = st.selectbox("Language",["English","Japanese"])
status = st.selectbox("Status",["Ongoing","Completed"])

if st.button("Predict"):

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

    st.success(f"Predicted Age Rating: {prediction}")

genre_map = {
    "Action":0,
    "Comedy":1,
    "Drama":2,
    "Fantasy":3
}

country_map = {
    "USA":0,
    "Japan":1,
    "Korea":2
}

format_map = {
    "Print":0,
    "Digital":1
}

language_map = {
    "English":0,
    "Japanese":1
}

status_map = {
    "Ongoing":0,
    "Completed":1
}
