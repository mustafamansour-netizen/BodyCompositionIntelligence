from __future__ import annotations

from datetime import date
import inspect

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from report_engine import (
    Profile,
    ReferenceBands,
    build_report_data,
    default_bands,
    html_to_pdf_bytes,
    load_withings_csvs,
    load_withings_zip,
    render_report_html,
)

st.set_page_config(page_title="Withings Body Composition Report", page_icon="📊", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1500px;}
[data-testid="stSidebar"] {min-width: 325px; max-width: 390px;}
.report-note {font-size:.9rem;color:#60717b;margin-top:-.35rem;margin-bottom:.8rem}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Withings → InBody-style Body Composition Report")
st.caption("Build V6.2")
st.markdown(
    "<div class='report-note'>Upload the original Withings export ZIP (recommended) or weight.csv + other.csv. "
    "The report uses the latest complete whole-body scan and the latest self-contained segmental snapshot; it never invents missing history values.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("1. Withings export")
    source_mode = st.radio("Import method", ["Withings ZIP", "Two CSV files"], horizontal=True)
    weight_df = other_df = None
    import_meta = {}
    load_error = None

    if source_mode == "Withings ZIP":
        up = st.file_uploader("Withings export ZIP", type=["zip"])
        if up is not None:
            try:
                weight_df, other_df, import_meta = load_withings_zip(up.getvalue())
            except Exception as exc:
                load_error = str(exc)
    else:
        w = st.file_uploader("weight.csv", type=["csv"], key="weight")
        o = st.file_uploader("other.csv", type=["csv"], key="other")
        if w is not None and o is not None:
            try:
                weight_df, other_df, import_meta = load_withings_csvs(w.getvalue(), o.getvalue())
            except Exception as exc:
                load_error = str(exc)

    if load_error:
        st.error(load_error)

    st.header("2. Profile")
    name = st.text_input("Name", value="")
    profile_id = st.text_input("Client / Profile ID", value="", help="Optional identifier shown in the report header for quicker practitioner recognition.")
    sex = st.selectbox("Sex", ["Male", "Female"])
    height_cm = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=182.0, step=0.5)
    use_dob = st.checkbox("Use date of birth", value=False)
    if use_dob:
        dob = st.date_input("Date of birth", value=date(1982, 1, 1), min_value=date(1920,1,1), max_value=date.today())
        age_override = None
    else:
        dob = None
        age_override = int(st.number_input("Age", min_value=18, max_value=100, value=43, step=1))

    use_goal = st.checkbox("Show personal goal", value=True)
    goal = st.number_input("Goal weight (kg)", min_value=35.0, max_value=200.0, value=82.0, step=0.5, disabled=not use_goal)

    st.markdown("#### History display")
    history_daily_rule = st.selectbox(
        "Daily history reading",
        ["Earliest complete scan", "Latest complete scan", "Daily median"],
        index=0,
        help="Chooses the one value shown for each measured day. Monthly medians are always calculated from all complete scans.",
    )
    history_points = st.select_slider(
        "History points shown",
        options=[6, 8, 10, 12],
        value=8,
        help="Number of measured days shown in the compact Body Composition History panel.",
    )

    base = default_bands(sex)
    with st.expander("Reference bands", expanded=False):
        st.caption("Defaults reproduce the reference bands used in the approved report. Adjust here if your Withings reference profile differs.")
        bf = st.slider("Body fat %", 5.0, 50.0, base.body_fat, 0.5)
        mm = st.slider("Muscle %", 40.0, 95.0, base.muscle_pct, 0.5)
        water = st.slider("Body water %", 30.0, 75.0, base.water_pct, 0.5)
        bone = st.slider("Bone mass %", 1.0, 8.0, base.bone_pct, 0.1)
        visceral = st.slider("Visceral fat index", 0.0, 20.0, base.visceral_fat, 0.5)
        bmi = st.slider("BMI", 14.0, 35.0, base.bmi, 0.1)

    bands = ReferenceBands(body_fat=bf, muscle_pct=mm, water_pct=water, bone_pct=bone, visceral_fat=visceral, bmi=bmi)

if weight_df is None or other_df is None:
    st.info("Upload the Withings export to generate the report.")
    st.markdown(
        """
### What this rebuild fixes
- No hard-coded body values or placeholder emoji model.
- Latest **complete** composition scan is used for the headline metrics.
- Segmental values come from one real segmental snapshot; dates are shown separately.
- History is based on actual CSV readings and **12 calendar months of monthly medians** with genuine gaps left blank.
- Goal body-fat math is a projection that assumes fat-free mass is maintained; it is not a prescription.
- Withings muscle mass is kept distinct from proprietary InBody SMM/score fields.
"""
    )
    st.stop()

profile_kwargs = {
    "name": name or "Profile",
    "sex": sex,
    "height_m": float(height_cm) / 100.0,
    "birth_date": dob,
    "age_override": age_override,
    "goal_weight_kg": float(goal) if use_goal else None,
    "history_daily_rule": history_daily_rule,
    "history_points": int(history_points),
    "profile_id": profile_id or None,
}

# Guard against a partial GitHub deployment where app.py was updated but
# report_engine.py is still from an older version. Streamlit otherwise shows a
# redacted TypeError that is not useful to the user.
profile_params = set(inspect.signature(Profile).parameters)
unsupported_profile_fields = [k for k in profile_kwargs if k not in profile_params]
if unsupported_profile_fields:
    st.error(
        "Deployment file mismatch: app.py is V5+ but report_engine.py is older. "
        "Replace report_engine.py from the same package, commit it, then reboot the Streamlit app. "
        f"Missing Profile fields: {', '.join(unsupported_profile_fields)}"
    )
    st.stop()

profile = Profile(**profile_kwargs)

try:
    report = build_report_data(weight_df, other_df, profile, bands=bands, diagnostics=import_meta)
except Exception as exc:
    st.error(str(exc))
    st.stop()

html = render_report_html(report)

c1, c2, c3, c4 = st.columns([1,1,1,2])
with c1:
    st.metric("Whole-body scan", report.scan_date.strftime("%d %b %Y"))
with c2:
    st.metric("Complete scans", str(report.diagnostics.get("complete_scans", "-")))
with c3:
    st.metric("Segment snapshot", report.segment_date.strftime("%d %b %Y") if report.segment_date is not None else "None")
with c4:
    st.caption("Report preview is rendered from the same HTML used for PDF export, so the two stay aligned.")

with st.expander("Data diagnostics", expanded=False):
    d1, d2, d3 = st.columns(3)
    d1.metric("weight.csv records", import_meta.get("weight_records", 0))
    d2.metric("other.csv records", import_meta.get("other_records", 0))
    d3.metric("Segment positions detected", report.diagnostics.get("segment_complete_positions", 0))
    st.write("Detected segment positions:", import_meta.get("positions", []))
    st.write("Detected other.csv metric types:")
    st.code("\n".join(import_meta.get("other_types", [])) or "No metric types detected")
    st.caption("If a Withings export changes a metric label, this list makes the mismatch visible instead of silently inventing a value.")

st.subheader("Report preview")
components.html(html, height=1750, scrolling=True)

st.subheader("Download")
col_a, col_b = st.columns(2)
with col_a:
    st.download_button(
        "Download printable HTML",
        data=html.encode("utf-8"),
        file_name=f"{profile.name.replace(' ','_')}_Withings_Body_Composition_Report.html",
        mime="text/html",
        use_container_width=True,
    )
with col_b:
    try:
        pdf_bytes = html_to_pdf_bytes(html)
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=f"{profile.name.replace(' ','_')}_Withings_Body_Composition_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as exc:
        st.warning(f"PDF export unavailable in this environment: {exc}. The HTML download remains print-ready.")
