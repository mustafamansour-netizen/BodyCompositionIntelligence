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

latest_scan = (
    complete_scans
    .sort_values("Date")
    .iloc[-1]
)

weight = float(
    latest_scan["Weight (kg)"]
)

fat_mass = float(
    latest_scan["Fat mass (kg)"]
)

bone_mass = float(
    latest_scan["Bone mass (kg)"]
)

muscle_mass = float(
    latest_scan["Muscle mass (kg)"]
)

hydration_mass = float(
    latest_scan["Hydration (kg)"]
)

fat_free_mass = weight - fat_mass

body_fat_pct = (
    fat_mass / weight
) * 100

muscle_pct = (
    muscle_mass / weight
) * 100

water_pct = (
    hydration_mass / weight
) * 100

height_m = 1.82

bmi = weight / (
    height_m ** 2
)

segment_types = [
    "Muscle Mass for segments",
    "Fat Mass for segments in mass unit",
    "Fat Free Mass for segments"
]

segment_df = other_df[
    other_df["type"].isin(
        segment_types
    )
].copy()

segment_df = segment_df.dropna(
    subset=[
        "date",
        "position",
        "value_numeric"
    ]
)

if segment_df.empty:
    latest_segment_date = None
    segment_table = pd.DataFrame()
else:
    latest_segment_date = (
        segment_df["date"].max()
    )

    latest_segment_scan = segment_df[
        segment_df["date"]
        == latest_segment_date
    ].copy()

    segment_table = (
        latest_segment_scan
        .pivot_table(
            index="position",
            columns="type",
            values="value_numeric",
            aggfunc="first"
        )
        .reset_index()
    )

def get_segment_value(
    position,
    metric
):

    if segment_table.empty:
        return None

    matching_rows = segment_table[
        segment_table["position"]
        == position
    ]

    if matching_rows.empty:
        return None

    if metric not in matching_rows.columns:
        return None

    value = matching_rows.iloc[0][
        metric
    ]

    if pd.isna(value):
        return None

    return float(value)

def format_kg(value):

    if value is None:
        return "No data"

    return f"{value:.1f} kg"

def calculate_difference(
    left_value,
    right_value
):

    if left_value is None:
        return None

    if right_value is None:
        return None

    average_value = (
        left_value + right_value
    ) / 2

    if average_value == 0:
        return None

    difference = (
        abs(
            left_value
            - right_value
        )
        / average_value
    ) * 100

    return difference

def difference_status(
    difference
):

    if difference is None:
        return "No data"

    if difference <= 5:
        return "Balanced"

    if difference <= 10:
        return "Monitor"

    return "Review"

left_arm_muscle = get_segment_value(
    "Left Arm",
    "Muscle Mass for segments"
)

left_arm_fat = get_segment_value(
    "Left Arm",
    "Fat Mass for segments in mass unit"
)

left_arm_ffm = get_segment_value(
    "Left Arm",
    "Fat Free Mass for segments"
)

right_arm_muscle = get_segment_value(
    "Right Arm",
    "Muscle Mass for segments"
)

right_arm_fat = get_segment_value(
    "Right Arm",
    "Fat Mass for segments in mass unit"
)

right_arm_ffm = get_segment_value(
    "Right Arm",
    "Fat Free Mass for segments"
)

left_leg_muscle = get_segment_value(
    "Left leg",
    "Muscle Mass for segments"
)

left_leg_fat = get_segment_value(
    "Left leg",
    "Fat Mass for segments in mass unit"
)

left_leg_ffm = get_segment_value(
    "Left leg",
    "Fat Free Mass for segments"
)

right_leg_muscle = get_segment_value(
    "Right leg",
    "Muscle Mass for segments"
)

right_leg_fat = get_segment_value(
    "Right leg",
    "Fat Mass for segments in mass unit"
)

right_leg_ffm = get_segment_value(
    "Right leg",
    "Fat Free Mass for segments"
)

torso_muscle = get_segment_value(
    "Torso",
    "Muscle Mass for segments"
)

torso_fat = get_segment_value(
    "Torso",
    "Fat Mass for segments in mass unit"
)

torso_ffm = get_segment_value(
    "Torso",
    "Fat Free Mass for segments"
)

arm_muscle_difference = (
    calculate_difference(
        left_arm_muscle,
        right_arm_muscle
    )
)

arm_fat_difference = (
    calculate_difference(
        left_arm_fat,
        right_arm_fat
    )
)

leg_muscle_difference = (
    calculate_difference(
        left_leg_muscle,
        right_leg_muscle
    )
)

leg_fat_difference = (
    calculate_difference(
        left_leg_fat,
        right_leg_fat
    )
)

st.success(
    "Whole-body scan: "
    + latest_scan["Date"].strftime(
        "%d %b %Y %H:%M:%S"
    )
)

if latest_segment_date is not None:
    st.info(
        "Segment scan: "
        + latest_segment_date.strftime(
            "%d %b %Y %H:%M:%S"
        )
    )
else:
    st.warning(
        "No valid segment scan was found."
    )

kpi1, kpi2, kpi3 = st.columns(3)

kpi1.metric(
    "Weight",
    f"{weight:.1f} kg"
)

kpi2.metric(
    "Body Fat",
    f"{body_fat_pct:.1f}%"
)

kpi3.metric(
    "Muscle Mass",
    f"{muscle_mass:.1f} kg"
)

kpi4, kpi5, kpi6 = st.columns(3)

kpi4.metric(
    "Water",
    f"{water_pct:.1f}%"
)

kpi5.metric(
    "BMI",
    f"{bmi:.1f}"
)

kpi6.metric(
    "Fat-Free Mass",
    f"{fat_free_mass:.1f} kg"
)

st.divider()

st.subheader(
    "Segmental Body Composition"
)

left_column, center_column, right_column = (
    st.columns(
        [2, 3, 2]
    )
)

with left_column:

    st.subheader("Left Arm")

    st.metric(
        "Muscle",
        format_kg(
            left_arm_muscle
        )
    )

    st.metric(
        "Fat",
        format_kg(
            left_arm_fat
        )
    )

    st.metric(
        "Fat-Free Mass",
        format_kg(
            left_arm_ffm
        )
    )

    st.divider()

    st.subheader("Left Leg")

    st.metric(
        "Muscle",
        format_kg(
            left_leg_muscle
        )
    )

    st.metric(
        "Fat",
        format_kg(
            left_leg_fat
        )
    )

    st.metric(
        "Fat-Free Mass",
        format_kg(
            left_leg_ffm
        )
    )

with center_column:

    st.markdown(
        """
        <div style="
            text-align: center;
            font-size: 170px;
            line-height: 1;
            padding-top: 25px;
            padding-bottom: 15px;
        ">
            &#129485;
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Torso")

    torso_col1, torso_col2, torso_col3 = (
        st.columns(3)
    )

    torso_col1.metric(
        "Muscle",
        format_kg(
            torso_muscle
        )
    )

    torso_col2.metric(
        "Fat",
        format_kg(
            torso_fat
        )
    )

    torso_col3.metric(
        "Fat-Free",
        format_kg(
            torso_ffm
        )
    )

with right_column:

    st.subheader("Right Arm")

    st.metric(
        "Muscle",
        format_kg(
            right_arm_muscle
        )
    )

    st.metric(
        "Fat",
        format_kg(
            right_arm_fat
        )
    )

    st.metric(
        "Fat-Free Mass",
        format_kg(
            right_arm_ffm
        )
    )

    st.divider()

    st.subheader("Right Leg")

    st.metric(
        "Muscle",
        format_kg(
            right_leg_muscle
        )
    )

    st.metric(
        "Fat",
        format_kg(
            right_leg_fat
        )
    )

    st.metric(
        "Fat-Free Mass",
        format_kg(
            right_leg_ffm
        )
    )

st.divider()

st.subheader(
    "Left and Right Balance"
)

balance1, balance2 = st.columns(2)

if arm_muscle_difference is None:
    arm_muscle_text = "No data"
else:
    arm_muscle_text = (
        f"{arm_muscle_difference:.1f}%"
    )

if arm_fat_difference is None:
    arm_fat_text = "No data"
else:
    arm_fat_text = (
        f"{arm_fat_difference:.1f}%"
    )

if leg_muscle_difference is None:
    leg_muscle_text = "No data"
else:
    leg_muscle_text = (
        f"{leg_muscle_difference:.1f}%"
    )

if leg_fat_difference is None:
    leg_fat_text = "No data"
else:
    leg_fat_text = (
        f"{leg_fat_difference:.1f}%"
    )

with balance1:

    st.markdown("### Arms")

    arm_col1, arm_col2 = (
        st.columns(2)
    )

    arm_col1.metric(
        "Muscle Difference",
        arm_muscle_text
    )

    arm_col2.metric(
        "Fat Difference",
        arm_fat_text
    )

    st.caption(
        "Muscle status: "
        + difference_status(
            arm_muscle_difference
        )
    )

with balance2:

    st.markdown("### Legs")

    leg_col1, leg_col2 = (
        st.columns(2)
    )

    leg_col1.metric(
        "Muscle Difference",
        leg_muscle_text
    )

    leg_col2.metric(
        "Fat Difference",
        leg_fat_text
    )

    st.caption(
        "Muscle status: "
        + difference_status(
            leg_muscle_difference
        )
    )

st.caption(
    "Balance difference is calculated as "
    "the absolute left-right difference "
    "divided by the average of both sides."
)

st.divider()

composition_column, assessment_column = (
    st.columns(
        [3, 2]
    )
)

with composition_column:

    st.subheader(
        "Body Composition Summary"
    )

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
                f"{body_fat_pct:.1f}%",
                f"{fat_mass:.2f} kg",
                f"{fat_free_mass:.2f} kg",
                f"{muscle_mass:.2f} kg",
                f"{muscle_pct:.1f}%",
                f"{bone_mass:.2f} kg",
                f"{hydration_mass:.2f} kg",
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

with assessment_column:

    st.subheader("Latest Assessment")

    st.metric(
        "Fat-Free Mass",
        f"{fat_free_mass:.1f} kg"
    )

    st.metric(
        "Bone Mass",
        f"{bone_mass:.1f} kg"
    )

    st.metric(
        "Hydration Mass",
        f"{hydration_mass:.1f} kg"
    )
