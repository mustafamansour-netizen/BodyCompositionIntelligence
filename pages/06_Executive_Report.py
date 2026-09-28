import streamlit as st

st.set_page_config(layout="wide")

st.title("📋 Executive Report")

st.markdown(
    """
    ## Body Composition Intelligence

    This page will become the final management dashboard.

    It will combine:

    ✅ Current Assessment

    ✅ Segment Analysis

    ✅ Historical Trends

    ✅ Goal Tracking

    ✅ Executive Summary

    Into a single screen.
    """
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Weight",
    "--"
)

col2.metric(
    "Body Fat %",
    "--"
)

col3.metric(
    "Muscle Mass",
    "--"
)

col4.metric(
    "BMI",
    "--"
)

st.divider()

st.subheader("Body Composition")

st.info(
    "This section will display body composition details."
)

st.divider()

st.subheader("Segment Analysis")

st.info(
    "This section will display arm, leg and torso analysis."
)

st.divider()

st.subheader("Historical Performance")

st.info(
    "This section will display trend charts."
)
