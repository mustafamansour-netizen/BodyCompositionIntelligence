import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Import Data",
    layout="wide"
)

st.title("Import Body Composition Data")

weight_file = st.file_uploader(
    "Upload weight.csv",
    type=["csv"],
    key="weight_upload"
)

other_file = st.file_uploader(
    "Upload other.csv",
    type=["csv"],
    key="other_upload"
)

if weight_file is not None and other_file is not None:

    weight_df = pd.read_csv(weight_file)
    other_df = pd.read_csv(other_file)

    required_weight_columns = [
        "Date",
        "Weight (kg)",
        "Fat mass (kg)",
        "Bone mass (kg)",
        "Muscle mass (kg)",
        "Hydration (kg)"
    ]

    required_other_columns = [
        "type",
        "date",
        "value",
        "unit",
        "position"
    ]

    missing_weight_columns = [
        column
        for column in required_weight_columns
        if column not in weight_df.columns
    ]

    missing_other_columns = [
        column
        for column in required_other_columns
        if column not in other_df.columns
    ]

    if missing_weight_columns:
        st.error(
            "Missing columns in weight.csv: "
            + ", ".join(missing_weight_columns)
        )
        st.stop()

    if missing_other_columns:
        st.error(
            "Missing columns in other.csv: "
            + ", ".join(missing_other_columns)
        )
        st.stop()

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

    weight_df = weight_df.dropna(
        subset=["Date", "Weight (kg)"]
    )

    other_df = other_df.dropna(
        subset=["date"]
    )

    st.session_state["weight_df"] = weight_df
    st.session_state["other_df"] = other_df

    complete_scans = weight_df.dropna(
        subset=[
            "Fat mass (kg)",
            "Bone mass (kg)",
            "Muscle mass (kg)",
            "Hydration (kg)"
        ]
    )

    weight_only_scans = weight_df[
        weight_df[
            [
                "Fat mass (kg)",
                "Bone mass (kg)",
                "Muscle mass (kg)",
                "Hydration (kg)"
            ]
        ].isna().all(axis=1)
    ]

    st.success(
        "Files loaded successfully. "
        "You can now use every other page without uploading again."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Weight Records",
        str(len(weight_df))
    )

    col2.metric(
        "Other Records",
        str(len(other_df))
    )

    col3.metric(
        "First Scan",
        weight_df["Date"].min().strftime("%d %b %Y")
    )

    col4.metric(
        "Latest Scan",
        weight_df["Date"].max().strftime("%d %b %Y")
    )

    st.divider()

    col5, col6, col7 = st.columns(3)

    col5.metric(
        "Complete Composition Scans",
        str(len(complete_scans))
    )

    col6.metric(
        "Weight-Only Scans",
        str(len(weight_only_scans))
    )

    col7.metric(
        "Detected Positions",
        str(other_df["position"].dropna().nunique())
    )

    st.divider()

    st.subheader("Latest Complete Scan")

    if complete_scans.empty:
        st.warning(
            "No complete body-composition scan was found."
        )
    else:
        latest_scan = (
            complete_scans
            .sort_values("Date")
            .tail(1)
        )

        st.dataframe(
            latest_scan,
            width="stretch",
            hide_index=True
        )

    st.divider()

    st.subheader("Segment Position Summary")

    segment_counts = (
        other_df["position"]
        .dropna()
        .value_counts()
        .rename_axis("Position")
        .reset_index(name="Record Count")
    )

    st.dataframe(
        segment_counts,
        width="stretch",
        hide_index=True
    )

else:

    if (
        "weight_df" in st.session_state
        and "other_df" in st.session_state
    ):
        st.success(
            "Data is already loaded in this session. "
            "You can use the other pages."
        )
    else:
        st.info(
            "Upload weight.csv and other.csv to begin."
        )
