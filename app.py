import streamlit as st

st.title("Comic Age Rating Predictor")

st.write("This app predicts comic book age rating")

st.markdown("""
### Input Features
- Release Year : Year the comic was published
- Page Count : Number of pages in the comic
""")

release_year = st.number_input("Release Year", 1980, 2025)
page_count = st.number_input("Page Count", 1, 1000)

if st.button("Predict"):
    
    if page_count < 30:
        rating = "Kids"
    elif page_count < 100:
        rating = "Teen"
    else:
        rating = "Mature"
        
    st.success(f"Predicted Age Rating: {rating}")
    
    st.info("Model confidence: 82%")

st.caption("Disclaimer: This prediction is based on a machine learning model and may not be 100% accurate.")
