import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Body Composition Data Import",
    layout="wide"
)

st.title("📊 Body Composition Data Import")

weight_file = st.file_uploader(
    "Upload weight.csv",
    type=["csv"]
)

other_file = st.file_uploader(
    "Upload other.csv",
    type=["csv"]
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

    weight_df = weight_df.dropna(
        subset=["Date", "Weight (kg)"]
    )

    other_df = other_df.dropna(
        subset=["date"]
    )

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

    st.success("Files loaded successfully")

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
        weight_df["Date"]
        .min()
        .strftime("%d %b %Y")
    )

    col4.metric(
        "Latest Scan",
        weight_df["Date"]
        .max()
        .strftime("%d %b %Y")
    )

    st.divider()

    scan_col1, scan_col2, scan_col3 = st.columns(3)

    scan_col1.metric(
        "Complete Composition Scans",
        str(len(complete_scans))
    )

    scan_col2.metric(
        "Weight-Only Scans",
        str(len(weight_only_scans))
    )

    scan_col3.metric(
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
            use_container_width=True,
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

    if segment_counts.empty:
        st.warning(
            "No segment positions were detected in other.csv."
        )
    else:
        st.dataframe(
            segment_counts,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    with st.expander("Preview weight.csv"):
        st.dataframe(
            weight_df.head(10),
            use_container_width=True,
            hide_index=True
        )

    with st.expander("Preview other.csv"):
        st.dataframe(
            other_df.head(10),
            use_container_width=True,
            hide_index=True
        )

else:
    st.info(
        "Upload both weight.csv and other.csv to begin."
    )
