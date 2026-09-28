import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Body Model",
    layout="wide"
)

st.title("Segment Body Analysis")

if "other_df" not in st.session_state:
    st.warning(
        "Open the Import page and upload both CSV files first."
    )
    st.stop()

other_df = st.session_state["other_df"].copy()

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

if segment_df.empty:
    st.error(
        "No recognized segment measurements were found."
    )
    st.stop()

latest_date = segment_df["date"].max()

latest_scan = segment_df[
    segment_df["date"] == latest_date
].copy()

pivot_table = (
    latest_scan
    .pivot_table(
        index="position",
        columns="type",
        values="value_numeric",
        aggfunc="first"
    )
    .reset_index()
)

def get_value(position, metric):

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

def display_value(value):

    if value is None:
        return "No data"

    return f"{value:.1f} kg"

left_arm_muscle = get_value(
    "Left Arm",
    "Muscle Mass for segments"
)

left_arm_fat = get_value(
    "Left Arm",
    "Fat Mass for segments in mass unit"
)

left_arm_ffm = get_value(
    "Left Arm",
    "Fat Free Mass for segments"
)

right_arm_muscle = get_value(
    "Right Arm",
    "Muscle Mass for segments"
)

right_arm_fat = get_value(
    "Right Arm",
    "Fat Mass for segments in mass unit"
)

right_arm_ffm = get_value(
    "Right Arm",
    "Fat Free Mass for segments"
)

left_leg_muscle = get_value(
    "Left leg",
    "Muscle Mass for segments"
)

left_leg_fat = get_value(
    "Left leg",
    "Fat Mass for segments in mass unit"
)

left_leg_ffm = get_value(
    "Left leg",
    "Fat Free Mass for segments"
)

right_leg_muscle = get_value(
    "Right leg",
    "Muscle Mass for segments"
)

right_leg_fat = get_value(
    "Right leg",
    "Fat Mass for segments in mass unit"
)

right_leg_ffm = get_value(
    "Right leg",
    "Fat Free Mass for segments"
)

torso_muscle = get_value(
    "Torso",
    "Muscle Mass for segments"
)

torso_fat = get_value(
    "Torso",
    "Fat Mass for segments in mass unit"
)

torso_ffm = get_value(
    "Torso",
    "Fat Free Mass for segments"
)

st.success(
    "Latest segment scan: "
    + latest_date.strftime("%d %b %Y %H:%M:%S")
)

left_column, body_column, right_column = st.columns(
    [2, 3, 2]
)

with left_column:

    st.subheader("Left Arm")

    st.metric(
        "Muscle",
        display_value(left_arm_muscle)
    )

    st.metric(
        "Fat",
        display_value(left_arm_fat)
    )

    st.metric(
        "Fat-Free",
        display_value(left_arm_ffm)
    )

    st.divider()

    st.subheader("Left Leg")

    st.metric(
        "Muscle",
        display_value(left_leg_muscle)
    )

    st.metric(
        "Fat",
        display_value(left_leg_fat)
    )

    st.metric(
        "Fat-Free",
        display_value(left_leg_ffm)
    )

with body_column:

    st.markdown("## Body Map")

    st.markdown(
        """
        <div style="
            text-align: center;
            font-size: 180px;
            line-height: 1;
            padding-top: 40px;
        ">
        &#129485;
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Torso")

    torso_col1, torso_col2, torso_col3 = st.columns(3)

    torso_col1.metric(
        "Muscle",
        display_value(torso_muscle)
    )

    torso_col2.metric(
        "Fat",
        display_value(torso_fat)
    )

    torso_col3.metric(
        "Fat-Free",
        display_value(torso_ffm)
    )

with right_column:

    st.subheader("Right Arm")

    st.metric(
        "Muscle",
        display_value(right_arm_muscle)
    )

    st.metric(
        "Fat",
        display_value(right_arm_fat)
    )

    st.metric(
        "Fat-Free",
        display_value(right_arm_ffm)
    )

    st.divider()

    st.subheader("Right Leg")

    st.metric(
        "Muscle",
        display_value(right_leg_muscle)
    )

    st.metric(
        "Fat",
        display_value(right_leg_fat)
    )

    st.metric(
        "Fat-Free",
        display_value(right_leg_ffm)
    )

st.divider()

arm_difference = None
leg_difference = None

if (
    left_arm_muscle is not None
    and right_arm_muscle is not None
    and left_arm_muscle + right_arm_muscle > 0
):
    arm_difference = (
        abs(left_arm_muscle - right_arm_muscle)
        / (
            (left_arm_muscle + right_arm_muscle)
            / 2
        )
        * 100
    )

if (
    left_leg_muscle is not None
    and right_leg_muscle is not None
    and left_leg_muscle + right_leg_muscle > 0
):
    leg_difference = (
        abs(left_leg_muscle - right_leg_muscle)
        / (
            (left_leg_muscle + right_leg_muscle)
            / 2
        )
        * 100
    )

balance_col1, balance_col2 = st.columns(2)

balance_col1.metric(
    "Arm Muscle Difference",
    "No data"
    if arm_difference is None
    else f"{arm_difference:.1f}%"
)

balance_col2.metric(
    "Leg Muscle Difference",
    "No data"
    if leg_difference is None
    else f"{leg_difference:.1f}%"
)
