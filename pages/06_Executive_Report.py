import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Executive Report",
    layout="wide"
)

st.title("Executive Body Composition Report")

if (
    "weight_df" not in st.session_state
    or "other_df" not in st.session_state
):
    st.warning(
        "Open the Import page and upload both CSV files first."
    )
    st.stop()

weight_df = st.session_state["weight_df"].copy()
other_df = st.session_state["other_df"].copy()

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
water_pct = hydration / weight * 100

height_m = 1.82
bmi = weight / (height_m ** 2)

segment_types = [
    "Muscle Mass for segments",
    "Fat Mass for segments in mass unit",
    "Fat Free Mass for segments"
]

segment_df = other_df[
    other_df["type"].isin(segment_types)
].copy()

segment_df = segment_df.dropna(
    subset=[
        "date",
        "position",
        "value_numeric"
    ]
)

latest_segment_date = None
pivot_table = pd.DataFrame()

if not segment_df.empty:

    latest_segment_date = segment_df["date"].max()

    latest_segment_scan = segment_df[
        segment_df["date"] == latest_segment_date
    ]

    pivot_table = (
        latest_segment_scan
        .pivot_table(
            index="position",
            columns="type",
            values="value_numeric",
            aggfunc="first"
        )
        .reset_index()
    )

def segment_value(position, metric):

    if pivot_table.empty:
        return None

    rows = pivot_table[
        pivot_table["position"] == position
    ]

    if rows.empty:
        return None

    if metric not in rows.columns:
        return None

    value = rows.iloc[0][metric]

    if pd.isna(value):
        return None

    return float(value)

def show_value(value):

    if value is None:
        return "No data"

    return f"{value:.1f} kg"

left_arm = segment_value(
    "Left Arm",
    "Muscle Mass for segments"
)

right_arm = segment_value(
    "Right Arm",
    "Muscle Mass for segments"
)

left_leg = segment_value(
    "Left leg",
    "Muscle Mass for segments"
)

right_leg = segment_value(
    "Right leg",
    "Muscle Mass for segments"
)

torso = segment_value(
    "Torso",
    "Muscle Mass for segments"
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

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric(
    "Weight",
    f"{weight:.1f} kg"
)

col2.metric(
    "Body Fat",
    f"{fat_pct:.1f}%"
)

col3.metric(
    "Fat Mass",
    f"{fat:.1f} kg"
)

col4.metric(
    "Muscle Mass",
    f"{muscle:.1f} kg"
)

col5.metric(
    "Water",
    f"{water_pct:.1f}%"
)

col6.metric(
    "BMI",
    f"{bmi:.1f}"
)

st.divider()

left_column, body_column, right_column = st.columns(
    [2, 3, 2]
)

with left_column:

    st.subheader("Left Side")

    st.metric(
        "Arm Muscle",
        show_value(left_arm)
    )

    st.metric(
        "Leg Muscle",
        show_value(left_leg)
    )

with body_column:

    st.markdown("## Segment Overview")

    st.markdown(
        """
        <div style="
            text-align: center;
            font-size: 170px;
            line-height: 1;
            padding-top: 20px;
        ">
        &#129485;
        </div>
        """,
        unsafe_allow_html=True
    )

    st.metric(
        "Torso Muscle",
        show_value(torso)
    )

with right_column:

    st.subheader("Right Side")

    st.metric(
        "Arm Muscle",
        show_value(right_arm)
    )

    st.metric(
        "Leg Muscle",
        show_value(right_leg)
    )

st.divider()

composition_column, assessment_column = st.columns(
    [3, 2]
)

with composition_column:

    st.subheader("Body Composition")

    summary = pd.DataFrame(
        {
            "Metric": [
                "Weight",
                "Fat Mass",
                "Fat-Free Mass",
                "Muscle Mass",
                "Bone Mass",
                "Hydration"
            ],
            "Value": [
                f"{weight:.2f} kg",
                f"{fat:.2f} kg",
                f"{fat_free:.2f} kg",
                f"{muscle:.2f} kg",
                f"{bone:.2f} kg",
                f"{hydration:.2f} kg"
            ]
        }
    )

    st.dataframe(
        summary,
        width="stretch",
        hide_index=True
    )

with assessment_column:

    st.subheader("Assessment")

    st.metric(
        "Fat-Free Mass",
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
