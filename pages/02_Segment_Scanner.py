import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Segment Scanner",
    layout="wide"
)

st.title("Segment Scanner")

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

wanted_positions = [
    "Left Arm",
    "Right Arm",
    "Torso",
    "Left leg",
    "Right leg"
]

pivot_table = pivot_table[
    pivot_table["position"].isin(wanted_positions)
].copy()

position_order = {
    "Left Arm": 1,
    "Right Arm": 2,
    "Torso": 3,
    "Left leg": 4,
    "Right leg": 5
}

pivot_table["sort_order"] = (
    pivot_table["position"]
    .map(position_order)
)

pivot_table = (
    pivot_table
    .sort_values("sort_order")
    .drop(columns=["sort_order"])
)

st.success(
    "Latest segment scan: "
    + latest_date.strftime("%d %b %Y %H:%M:%S")
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Segment Records",
    str(len(segment_df))
)

col2.metric(
    "Segment Scan Timestamps",
    str(segment_df["date"].nunique())
)

col3.metric(
    "Positions in Latest Scan",
    str(pivot_table["position"].nunique())
)

st.divider()

st.subheader("Latest Segment Measurements")

st.dataframe(
    pivot_table,
    width="stretch",
    hide_index=True
)

st.divider()

st.subheader("Available Segment Scan Dates")

scan_dates = (
    segment_df["date"]
    .drop_duplicates()
    .sort_values(ascending=False)
    .to_frame(name="Segment Scan Date")
)

st.dataframe(
    scan_dates.head(20),
    width="stretch",
    hide_index=True
)
``
