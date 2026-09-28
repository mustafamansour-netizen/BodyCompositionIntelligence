import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Historical Trends",
    layout="wide"
)

st.title("Historical Trends")

if "weight_df" not in st.session_state:
    st.warning(
        "Open the Import page and upload both CSV files first."
    )
    st.stop()

weight_df = st.session_state["weight_df"].copy()

weight_df = weight_df.dropna(
    subset=[
        "Date",
        "Weight (kg)"
    ]
)

weight_df = weight_df.sort_values("Date")

weight_df["Fat %"] = (
    weight_df["Fat mass (kg)"]
    / weight_df["Weight (kg)"]
    * 100
)

weight_df["Muscle %"] = (
    weight_df["Muscle mass (kg)"]
    / weight_df["Weight (kg)"]
    * 100
)

weight_df["Water %"] = (
    weight_df["Hydration (kg)"]
    / weight_df["Weight (kg)"]
    * 100
)

available_metrics = [
    "Weight (kg)",
    "Fat mass (kg)",
    "Muscle mass (kg)",
    "Hydration (kg)",
    "Fat %",
    "Muscle %",
    "Water %"
]

metric = st.selectbox(
    "Choose metric",
    available_metrics
)

aggregation = st.selectbox(
    "Choose aggregation",
    [
        "Individual Scans",
        "Daily Median",
        "Weekly Median",
        "Monthly Median"
    ]
)

chart_df = weight_df[
    [
        "Date",
        metric
    ]
].dropna()

if aggregation == "Daily Median":
    chart_df = (
        chart_df
        .set_index("Date")
        .resample("D")
        .median()
        .dropna()
        .reset_index()
    )

elif aggregation == "Weekly Median":
    chart_df = (
        chart_df
        .set_index("Date")
        .resample("W")
        .median()
        .dropna()
        .reset_index()
    )

elif aggregation == "Monthly Median":
    chart_df = (
        chart_df
        .set_index("Date")
        .resample("MS")
        .median()
        .dropna()
        .reset_index()
    )

fig = px.line(
    chart_df,
    x="Date",
    y=metric,
    markers=True,
    title=metric
)

fig.update_layout(
    height=560,
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    )
)

st.plotly_chart(
    fig,
    width="stretch"
)

latest_complete = (
    weight_df
    .dropna(
        subset=[
            "Fat mass (kg)",
            "Muscle mass (kg)",
            "Hydration (kg)"
        ]
    )
    .sort_values("Date")
)

if not latest_complete.empty:

    latest = latest_complete.iloc[-1]

    latest_fat_pct = (
        latest["Fat mass (kg)"]
        / latest["Weight (kg)"]
        * 100
    )

    latest_muscle_pct = (
        latest["Muscle mass (kg)"]
        / latest["Weight (kg)"]
        * 100
    )

    latest_water_pct = (
        latest["Hydration (kg)"]
        / latest["Weight (kg)"]
        * 100
    )

    st.subheader("Latest Complete Reading")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Weight",
        f"{latest['Weight (kg)']:.1f} kg"
    )

    col2.metric(
        "Body Fat",
        f"{latest_fat_pct:.1f}%"
    )

    col3.metric(
        "Muscle",
        f"{latest_muscle_pct:.1f}%"
    )

    col4.metric(
        "Water",
        f"{latest_water_pct:.1f}%"
    )
