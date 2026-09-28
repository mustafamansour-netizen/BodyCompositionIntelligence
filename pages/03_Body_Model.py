import streamlit as st
import pandas as pd

st.title("Body Model")

other_file = st.file_uploader(
    "Upload other.csv",
    type=["csv"]
)

if other_file:

    st.success("File loaded")

    other_df = pd.read_csv(other_file)

    st.write(other_df.head())
