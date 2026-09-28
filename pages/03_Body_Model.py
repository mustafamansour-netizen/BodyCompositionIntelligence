import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.title("🧍 Body Model")

other_file = st.file_uploader(
    "Upload other.csv",
    type=["csv"]
)

if other_file:

    other_df = pd.read_csv(other_file)

    other_df["date"] = pd.to_datetime(
        other_df["date"],
        errors="coerce"
    )

    segment_types = [
        "Muscle Mass for segments",
        "Fat Mass for segments in mass unit",
        "Fat Free Mass for segments"
    ]

    segment_df = other_df[
        other_df["type"].isin(segment_types)
    ]

    latest_date = segment_df["date"].max()

    latest_scan = segment_df[
        segment_df["date"] == latest_date
    ]

    pivot_table = (
        latest_scan
        .pivot_table(
            index="position",
            columns="type",
            values="value",
            aggfunc="first"
        )
        .reset_index()
    )

    st.success(
        f"Latest Segment Scan: {latest_date}"
    )

    st.dataframe(
        pivot_table,
        use_container_width=True
    )
