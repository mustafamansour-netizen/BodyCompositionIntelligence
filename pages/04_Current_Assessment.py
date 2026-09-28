import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.title("📊 Current Assessment")

if "weight_df" not in st.session_state:
 
st.warning(
"Please load files in Import page first."
)
 
st.stop()
 
weight_df = st.session_state["weight_df"]

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
        f"{weight:.1f} kg"
    )

    col2.metric(
        "Body Fat %",
        f"{fat_pct:.1f}%"
    )

    col3.metric(
        "Fat Mass",
        f"{fat:.1f} kg"
    )

    col4.metric(
        "Muscle",
        f"{muscle:.1f} kg"
    )

    col5.metric(
        "Water %",
        f"{water_pct:.1f}%"
    )

    col6.metric(
        "BMI",
        f"{bmi:.1f}"
    )

    st.divider()

    st.subheader("Current Body Composition")

    summary = pd.DataFrame(
        {
            "Metric": [
                "Weight",
                "Fat Mass",
                "Fat Free Mass",
                "Muscle Mass",
                "Bone Mass",
                "Hydration"
            ],
            "Value": [
                round(weight, 2),
                round(fat, 2),
                round(fat_free, 2),
                round(muscle, 2),
                round(bone, 2),
                round(hydration, 2)
            ]
        }
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "Upload weight.csv to display the latest assessment."
    )
