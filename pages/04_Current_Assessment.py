import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Current Assessment",
    layout="wide"
)

st.title("Current Assessment")

if "weight_df" not in st.session_state:
    st.warning(
        "Open the Import page and upload both CSV files first."
    )
    st.stop()

weight_df = st.session_state["weight_df"].copy()

complete_scans = weight_df.dropna(
    subset=[
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

latest_scan = (
    complete_scans
    .sort_values("Date")
    .iloc[-1]
)

weight = float(latest_scan["Weight (kg)"])
fat = float(latest_scan["Fat mass (kg)"])
bone = float(latest_scan["Bone mass (kg)"])
muscle = float(latest_scan["Muscle mass (kg)"])
hydration = float(latest_scan["Hydration (kg)"])

fat_free = weight - fat
fat_pct = fat / weight * 100
muscle_pct = muscle / weight * 100
water_pct = hydration / weight * 100

height_m = 1.82
bmi = weight / (height_m ** 2)

st.success(
    "Latest complete scan: "
    + latest_scan["Date"].strftime(
        "%d %b %Y %H:%M:%S"
    )
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Weight",
    f"{weight:.1f} kg"
)

col2.metric(
    "Body Fat",
    f"{fat_pct:.1f}%"
)

col3.metric(
    "Muscle Mass",
    f"{muscle:.1f} kg"
)

col4.metric(
    "BMI",
    f"{bmi:.1f}"
)

col5, col6, col7, col8 = st.columns(4)

col5.metric(
    "Fat Mass",
    f"{fat:.1f} kg"
)

col6.metric(
    "Fat-Free Mass",
    f"{fat_free:.1f} kg"
)

col7.metric(
    "Water",
    f"{water_pct:.1f}%"
)

col8.metric(
    "Bone Mass",
    f"{bone:.1f} kg"
)

st.divider()

st.subheader("Body Composition Summary")

summary = pd.DataFrame(
    {
        "Metric": [
            "Weight",
            "Body Fat Percentage",
            "Fat Mass",
            "Fat-Free Mass",
            "Muscle Mass",
            "Muscle Percentage",
            "Bone Mass",
            "Hydration Mass",
            "Water Percentage",
            "BMI"
        ],
        "Value": [
            f"{weight:.2f} kg",
            f"{fat_pct:.1f}%",
            f"{fat:.2f} kg",
            f"{fat_free:.2f} kg",
            f"{muscle:.2f} kg",
            f"{muscle_pct:.1f}%",
            f"{bone:.2f} kg",
            f"{hydration:.2f} kg",
            f"{water_pct:.1f}%",
            f"{bmi:.1f}"
        ]
    }
)

st.dataframe(
    summary,
    width="stretch",
    hide_index=True
)
