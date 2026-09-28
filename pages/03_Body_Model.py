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

required_columns = [
    "type",
    "date",
    "value",
    "position"
]

missing_columns = [
    column
    for column in required_columns
    if column not in other_df.columns
]

if missing_columns:
    st.error(
        "Missing columns in other.csv: "
        + ", ".join(missing_columns)
    )
    st.stop()

other_df["date"] = pd.to_datetime(
    other_df["date"],
    errors="coerce"
)

other_df["value_numeric"] = pd.to_numeric(
    other_df["value"],
    errors="coerce"
)

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
        "No valid segment measurements were found."
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

    matching_rows = pivot_table[
        pivot_table["position"] == position
    ]

    if matching_rows.empty:
        return None

    if metric not in matching_rows.columns:
        return None

    value = matching_rows.iloc[0][metric]

    if pd.isna(value):
        return None

    return float(value)

def format_value(value):

    if value is None:
        return "No data"

    return f"{value:.1f} kg"

def calculate_difference(left_value, right_value):

    if left_value is None or right_value is None:
        return None

    average_value = (
        left_value + right_value
