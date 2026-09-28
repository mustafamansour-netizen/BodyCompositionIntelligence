import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.title("🧍 Segment Body Analysis")

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

    left, center, right = st.columns([2,3,2])

    with left:

        st.subheader("Left Arm")

        st.dataframe(
            pivot_table[
                pivot_table["position"]=="Left Arm"
            ]
        )

        st.subheader("Left Leg")

        st.dataframe(
            pivot_table[
                pivot_table["position"]=="Left leg"
            ]
        )

    with center:

        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Human_body_silhouette.svg/512px-Human_body_silhouette.svg.png",
            width=300
        )

    with right:

        st.subheader("Right Arm")

        st.dataframe(
            pivot_table[
                pivot_table["position"]=="Right Arm"
            ]
        )

        st.subheader("Right Leg")

        st.dataframe(
            pivot_table[
                pivot_table["position"]=="Right leg"
            ]
        )

    st.subheader("Torso")

    st.dataframe(
        pivot_table[
            pivot_table["position"]=="Torso"
        ]
    )
