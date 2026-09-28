import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Executive Report",
    layout="wide"
)

st.title("Executive Body Composition Report")

if "weight_df" not in st.session_state:
    st.warning(
        "Open the Import page and upload both CSV files first."
    )
    st.stop()

if "other_df" not in st.session_state:
    st.warning(
        "Open the Import page and upload both CSV files first."
    )
    st.stop()

weight_df = st.session_state["weight_df"].copy()
other_df = st.session_state["other_df"].copy()

weight_df["Date"] = pd.to_datetime(
    weight_df["Date"],
    errors="coerce"
)

other_df["date"] = pd.to_datetime(
    other_df["date"],
    errors="coerce"
)

other_df["value_numeric"] = pd.to_numeric(
    other_df["value"],
    errors="coerce"
)

complete_scans = weight_df.dropna(
    subset=[
        "Date",
        "Weight (kg)",
        "Fat mass (kg)",
        "Bone mass (kg)",
        "Muscle mass (kg)",
        "Hydration (kg)"
    ]
)

if complete_scans.empty:
    st.error(
        "No complete body-composition scan was found."
    )
    st.stop()

latest_scan = 
