import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

st.title("📈 Historical Trends")

weight_file = st.file_uploader(
    "Upload weight.csv",
    type=["csv"]
)

if weight_file:

    weight_df = pd.read_csv(weight_file)

    weight_df["Date"] = pd.to_datetime(
        weight_df["Date"],
        errors="coerce"
    )

    weight_df = weight_df.sort_values(
        "Date"
    )

    weight_df["Fat %"] = (
        weight_df["Fat mass (kg)"]
        /
        weight_df["Weight (kg)"]
    ) * 100

    weight_df["Muscle %"] = (
        weight_df["Muscle mass (kg)"]
        /
        weight_df["Weight (kg)"]
    ) * 100

    st.success(
        f"{len(weight_df)} records loaded"
    )

    metric = st.selectbox(
        "Choose Metric",
        [
            "Weight (kg)",
            "Fat mass (kg)",
            "Muscle mass (kg)",
            "Hydration (kg)",
            "Fat %",
            "Muscle %"
        ]
    )

    fig = px.line(
        weight_df,
        x="Date",
        y=metric,
        markers=True,
        title=metric
    )

    fig.update_layout(
        height=600
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Latest Values")

    latest = weight_df.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Weight",
        f"{latest['Weight (kg)']:.1f} kg"
    )

    c2.metric(
        "Fat %",
        f"{latest['Fat %']:.1f}%"
    )

    c3.metric(
        "Muscle %",
        f"{latest['Muscle %']:.1f}%"
    )

    c4.metric(
        "Hydration",
        f"{latest['Hydration (kg)']:.1f} kg"
    )

else:

    st.info(
        "Upload weight.csv to view historical trends."
    )
