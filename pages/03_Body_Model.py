import streamlit as st

st.title("Body Model")

if "other_df" not in st.session_state:

    st.warning(
        "Load data from Import page first."
    )

    st.stop()

left_col, center_col, right_col = st.columns(
    [2, 3, 2]
)

with left_col:

    st.subheader("Left Arm")

    st.metric(
        "Muscle",
        "4.6 kg"
    )

    st.subheader("Left Leg")

    st.metric(
        "Muscle",
        "12.6 kg"
    )

with center_col:

    st.markdown(
        """
        <div style='text-align:center;font-size:180px'>
        🧍
        </div>
        """,
        unsafe_allow_html=True
    )

with right_col:

    st.subheader("Right Arm")

    st.metric(
        "Muscle",
        "4.8 kg"
    )

    st.subheader("Right Leg")

    st.metric(
        "Muscle",
        "12.5 kg"
    )
