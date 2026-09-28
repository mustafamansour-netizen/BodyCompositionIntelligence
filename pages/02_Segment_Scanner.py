import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.title("🔍 Segment Scanner")

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

    st.success("other.csv loaded")

    segment_types = [
        "Muscle Mass for segments",
        "Fat Mass for segments in mass unit",
        "Fat Free Mass for segments"
    ]

    segment_df = other_df[
        other_df["type"].isin(segment_types)
    ]

    st.metric(
        "Segment Records",
        len(segment_df)
    )

    st.subheader("Detected Segment Data")

    st.dataframe(
        segment_df[
            [
                "date",
                "type",
                "position",
                "value"
            ]
        ].head(100),
        use_container_width=True
    )

    st.subheader("Segment Positions")

    st.dataframe(
        segment_df["position"]
        .value_counts()
        .reset_index(),
        use_container_width=True
    )
