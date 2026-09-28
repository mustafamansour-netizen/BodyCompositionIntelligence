import streamlit as st
import pandas as pd

st.title("Import Body Composition Data")

weight_file = st.file_uploader(
    "Upload weight.csv",
    type=["csv"]
)

other_file = st.file_uploader(
    "Upload other.csv",
    type=["csv"]
)

if weight_file:
    
    weight_df = pd.read_csv(weight_file)

    st.success("weight.csv loaded")

    st.write(f"Rows: {len(weight_df)}")

    st.dataframe(weight_df.head())


if other_file:

    other_df = pd.read_csv(other_file)

    st.success("other.csv loaded")

    st.write(f"Rows: {len(other_df)}")

    st.dataframe(other_df.head())
