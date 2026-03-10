import streamlit as st
import pandas as pd

st.title("Comic Age Rating Predictor")

st.write("This app predicts comic book age rating")

release_year = st.number_input("Release Year", 1980, 2025)
page_count = st.number_input("Page Count", 1, 1000)

if st.button("Predict"):
    st.success("Predicted Age Rating: Teen")
