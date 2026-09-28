import streamlit as st
import pandas as pd

st.title("Segment Scanner")

if "other_df" not in st.session_state:

    st.warning(
        "Load data from Import page first."
    )

    st.stop()

other_df = st.session_state["other_df"]

segment_types = [
    "Muscle Mass for segments",
    "Fat Mass for segments in mass unit",
    "Fat Free Mass for segments"
]

segment_df = other_df[
    other_df["type"].isin(segment_types)
]

segment_df["value_numeric"] = pd.to_numeric(
    segment_df["value"],
    errors="coerce"
)

latest_date = segment_df["date"].max()

latest_scan = segment_df[
    segment_df["date"] == latest_date
]

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

st.success(
    f"Latest Segment Scan: {latest_date}"
)

st.dataframe(
    pivot_table,
    width="stretch"
)
