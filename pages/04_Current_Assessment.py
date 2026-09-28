import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.title("📊 Current Assessment")

weight_file = st.file_uploader(
    "Upload weight.csv",
    type=["csv"]
)

if weight_file:

    weight_df = pd.read_csv(weight_file)

    weight_df["Date"] = pd.to_datetime(
        weight_df["Date"],
        errors="coerce"
    )

    complete_scans = weight_df.dropna(
        subset=[
            "Fat mass (kg)",
            "Bone mass (kg)",
            "Muscle mass (kg)",
            "Hydration (kg)"
        ]
    )

    latest_scan = (
        complete_scans
        .sort_values("Date")
        .iloc[-1]
    )

    weight = latest_scan["Weight (kg)"]
    fat = latest_scan["Fat mass (kg)"]
    muscle = latest_scan["Muscle mass (kg)"]
    hydration = latest_scan["Hydration (kg)"]
    bone = latest_scan["Bone mass (kg)"]

    fat_free = weight - fat

    fat_pct = (fat / weight) * 100
    muscle_pct = (muscle / weight) * 100
    water_pct = (hydration / weight) * 100

    height_m = 1.82

    bmi = weight / (height_m ** 2)

    st.success(
        f"Latest Complete Scan: {latest_scan['Date']}"
    )

    st.divider()

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric(
        "Weight",
        f"{weight
