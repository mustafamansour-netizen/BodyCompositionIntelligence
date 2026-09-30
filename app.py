from __future__ import annotations

from datetime import date, timedelta
import inspect

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from report_engine import (
    ENGINE_BUILD,
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
st.caption("Report Engine V8.1")
st.markdown(
    "<div class='report-note'>Upload the original Withings export ZIP (recommended) or weight.csv + other.csv. "
    "V8.1 keeps the V8 universal unit/import engine and refines goal tracking so starting weight alone unlocks real progress, while journey date remains optional.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("1. Withings export")
    source_mode = st.radio("Import method", ["Withings ZIP", "Two CSV files"], horizontal=True)
    with st.expander("Advanced import settings", expanded=False):
        source_unit_override_ui = st.selectbox(
            "Source mass unit override", ["Auto-detect", "kg", "lb"], index=0,
            help="Leave on Auto-detect for normal Withings exports. Use kg/lb only when a non-standard export omits unit labels."
        )
    source_unit_override = None if source_unit_override_ui == "Auto-detect" else source_unit_override_ui
    weight_df = other_df = None
    import_meta = {}
    load_error = None

    if source_mode == "Withings ZIP":
        up = st.file_uploader("Withings export ZIP", type=["zip"])
        if up is not None:
            try:
                weight_df, other_df, import_meta = load_withings_zip(up.getvalue(), source_mass_unit_override=source_unit_override)
            except Exception as exc:
                load_error = str(exc)
    else:
        w = st.file_uploader("weight.csv", type=["csv"], key="weight")
        o = st.file_uploader("other.csv", type=["csv"], key="other")
        with st.expander("Optional Body Scan files", expanded=False):
            pwv = st.file_uploader("pwv.csv (optional)", type=["csv"], key="pwv")
            bp = st.file_uploader("bp.csv (optional, used for scan heart rate)", type=["csv"], key="bp")
        if w is not None and o is not None:
            try:
                weight_df, other_df, import_meta = load_withings_csvs(
                    w.getvalue(), o.getvalue(),
                    pwv.getvalue() if pwv is not None else None,
                    bp.getvalue() if bp is not None else None,
                    source_mass_unit_override=source_unit_override,
                )
            except Exception as exc:
                load_error = str(exc)

    if load_error:
        st.error(load_error)

    st.header("2. Profile")
    name = st.text_input("Name", value="")
    profile_id = st.text_input("Client / Profile ID", value="", help="Optional identifier shown in the report header.")
    sex = st.selectbox("Sex", ["Male", "Female"])
    display_units = st.selectbox("Report units", ["Metric", "US Customary"], index=0, help="Independent of the units used in the imported Withings ZIP.")
    if display_units == "Metric":
        height_cm = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=182.0, step=0.5)
        height_m = float(height_cm) / 100.0
    else:
        hc1, hc2 = st.columns(2)
        with hc1:
            height_ft = st.number_input("Height (ft)", min_value=3, max_value=7, value=5, step=1)
        with hc2:
            height_in = st.number_input("Height (in)", min_value=0.0, max_value=11.9, value=11.7, step=0.1)
        height_m = (float(height_ft) * 12.0 + float(height_in)) * 0.0254
    use_dob = st.checkbox("Use date of birth", value=False)
    if use_dob:
        dob = st.date_input("Date of birth", value=date(1982, 1, 1), min_value=date(1920,1,1), max_value=date.today())
        age_override = None
    else:
        dob = None
        age_override = int(st.number_input("Age", min_value=18, max_value=100, value=43, step=1))

    st.markdown("#### Goal settings")
    use_goal = st.checkbox("Show personal goal", value=True)
    if display_units == "Metric":
        goal_display = st.number_input("Goal weight (kg)", min_value=35.0, max_value=200.0, value=82.0, step=0.5, disabled=not use_goal)
        goal_kg = float(goal_display) if use_goal else None
    else:
        goal_display = st.number_input("Goal weight (lb)", min_value=77.0, max_value=440.0, value=180.8, step=1.0, disabled=not use_goal)
        goal_kg = float(goal_display) / 2.20462262185 if use_goal else None

    use_journey = st.checkbox(
        "Track journey progress", value=False, disabled=not use_goal,
        help="Starting weight is enough to calculate progress %. Journey start date is optional."
    )
    if use_journey and use_goal:
        if display_units == "Metric":
            starting_weight_display = st.number_input(
                "Starting weight (kg)", min_value=35.0, max_value=250.0, value=95.0, step=0.1,
                help="Unlocks real journey progress in the report.",
            )
            starting_weight = float(starting_weight_display)
        else:
            starting_weight_display = st.number_input(
                "Starting weight (lb)", min_value=77.0, max_value=550.0, value=209.4, step=0.5,
                help="Unlocks real journey progress in the report.",
            )
            starting_weight = float(starting_weight_display) / 2.20462262185
        use_journey_date = st.checkbox("Add journey start date", value=False)
        if use_journey_date:
            journey_start_date = st.date_input(
                "Journey start date", value=date.today()-timedelta(days=90), min_value=date(2000,1,1), max_value=date.today()
            )
        else:
            journey_start_date = None
    else:
        journey_start_date = None
        starting_weight = None

    use_bf_goal = st.checkbox("Set target body fat %", value=False, disabled=not use_goal)
    target_bf = st.number_input(
        "Target body fat (%)", min_value=5.0, max_value=50.0, value=15.0, step=0.5,
        disabled=not (use_goal and use_bf_goal),
        help="Optional composition target. The report treats this as a user-defined goal, not a prescribed clinical target.",
    )

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
    st.caption(f"Engine build: {ENGINE_BUILD}")

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
    "height_m": float(height_m),
    "birth_date": dob,
    "age_override": age_override,
    "goal_weight_kg": goal_kg,
    "history_daily_rule": history_daily_rule,
    "history_points": int(history_points),
    "profile_id": profile_id or None,
    "journey_start_date": journey_start_date,
    "starting_weight_kg": float(starting_weight) if starting_weight is not None else None,
    "target_body_fat_pct": float(target_bf) if use_bf_goal else None,
    "display_units": display_units,
}

# Guard against a partial GitHub deployment where app.py was updated but
# report_engine.py is still from an older version. Streamlit otherwise shows a
# redacted TypeError that is not useful to the user.
profile_params = set(inspect.signature(Profile).parameters)
unsupported_profile_fields = [k for k in profile_kwargs if k not in profile_params]
if unsupported_profile_fields:
    st.error(
        "Deployment file mismatch: app.py and report_engine.py are from different builds. "
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
    st.write("Import normalization:", import_meta.get("source_mass_columns", {}))
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
