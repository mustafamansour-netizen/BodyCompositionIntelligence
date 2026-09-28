import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.title("📊 Body Composition Data Import")

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

    # Convert dates
    weight_df["Date"] = pd.to_datetime(
        weight_df["Date"],
        errors="coerce"
    )

    other_df["date"] = pd.to_datetime(
        other_df["date"],
        errors="coerce"
    )

    st.success("Files loaded successfully")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Weight Records",
        len(weight_df)
    )

    col2.metric(
        "Other Records",
        len(other_df)
    )

    col3.metric(
        "First Scan",
        weight_df["Date"].min().date()
    )

    col4.metric(
        "Latest Scan",
        weight_df["Date"].max().date()
    )

    st.divider()

    # Complete scans
    complete_scans = weight_df.dropna(
        subset=[
            "Fat mass (kg)",
            "Bone mass (kg)",
            "Muscle mass (kg)",
            "Hydration (kg)"
        ]
    )

    st.metric(
        "Complete Body Composition Scans",
        len(complete_scans)
    )

    st.divider()

    st.subheader("Latest Complete Scan")

    latest_scan = complete_scans.sort_values(
        "Date"
    ).iloc[-1]

    st.dataframe(
        latest_scan.to_frame().T,
        use_container_width=True
    )

    st.divider()

    st.subheader("Detected Segment Positions")

    segment_positions = sorted(
        other_df["position"]
        .dropna()
        .unique()
    )

    st.write(segment_positions)

    st.divider()

    st.subheader("Preview: weight.csv")

    st.dataframe(
        weight_df.head(10),
        use_container_width=True
    )

    st.subheader("Preview: other.csv")

    st.dataframe(
        other_df.head(10),
        use_container_width=True
    )
