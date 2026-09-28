import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.title("📂 Import Data")

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

    weight_df["Date"] = pd.to_datetime(
        weight_df["Date"],
        errors="coerce"
    )

    other_df["date"] = pd.to_datetime(
        other_df["date"],
        errors="coerce"
    )

    st.session_state["weight_df"] = weight_df
    st.session_state["other_df"] = other_df

    st.success(
        "Files loaded and stored in memory."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Weight Records",
            len(weight_df)
        )

    with col2:
        st.metric(
            "Other Records",
            len(other_df)
        )

else:

    st.info(
        "Upload both files to begin."
    )
