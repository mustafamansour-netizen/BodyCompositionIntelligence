import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Executive Report",
    layout="wide"
)

st.title("📋 Executive Report")

weight_file = st.file_uploader(
    "Upload weight.csv",
    type=["csv"]
)

other_file = st.file_uploader(
    "Upload other.csv",
    type=["csv"]
)

if weight_file and other_file:

    weight_df = pd.read_csv(weight_file)
    other_df = pd.read_csv(other_file)

    weight_df["Date"] = pd.to_datetime(
        weight_df["Date"],
        errors="coerce"
    )

    other_df["date"] = pd.to_datetime(
        other_df["date"],
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
    water_pct = (hydration / weight) * 100

    bmi = weight / (1.82 ** 2)

    st.success(
        f"Latest Scan: {latest_scan['Date']}"
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    k1.metric(
        "Weight",
        f"{weight:.1f} kg"
    )

    k2.metric(
        "Body Fat %",
        f"{fat_pct:.1f}%"
    )

    k3.metric(
        "Fat Mass",
        f"{fat:.1f} kg"
    )

    k4.metric(
        "Muscle",
        f"{muscle:.1f} kg"
    )

    k5.metric(
        "Water %",
        f"{water_pct:.1f}%"
    )

    k6.metric(
        "BMI",
        f"{bmi:.1f}"
    )

    st.divider()

    left, center, right = st.columns([2,2,2])

    with left:

        st.subheader("💪 Left Side")

        st.metric(
            "Arm Muscle",
            "4.6 kg"
        )

        st.metric(
            "Leg Muscle",
            "12.6 kg"
        )

    with center:

        st.markdown(
            """
            <div style="
                text-align:center;
                font-size:180px;
            ">
            🧍
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:

        st.subheader("💪 Right Side")

        st.metric(
            "Arm Muscle",
            "4.8 kg"
        )

        st.metric(
            "Leg Muscle",
            "12.5 kg"
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Body Composition")

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

    with col2:

        st.subheader("Assessment")

        st.metric(
            "Fat Free Mass",
            f"{fat_free:.1f} kg"
        )

        st.metric(
            "Bone Mass",
            f"{bone:.1f} kg"
        )

        st.metric(
            "Hydration",
            f"{hydration:.1f} kg"
        )

else:

    st.info(
        "Upload weight.csv and other.csv"
    )
