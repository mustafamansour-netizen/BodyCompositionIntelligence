from datetime import date
import pandas as pd

from report_engine import Profile, build_report_data, normalize_frames, render_report_html


def synthetic_frames():
    rows = [
        ["2026-09-17 07:10:00",92.2,19.73,3.5,69.0,49.0],
        ["2026-09-18 07:12:00",92.5,18.78,3.5,70.1,49.2],
        ["2026-09-19 07:14:00",92.4,19.22,3.5,69.6,49.1],
        ["2026-09-20 07:16:00",92.6,20.37,3.5,68.8,49.0],
        ["2026-09-21 07:18:00",92.6,20.09,3.5,69.0,49.1],
        ["2026-09-22 07:00:00",92.30,19.00,3.52,69.40,49.00],
        ["2026-09-22 10:26:00",91.94,18.43,3.53,69.98,49.28],
    ]
    weight = pd.DataFrame(rows, columns=[
        "Date","Weight (kg)","Fat mass (kg)","Bone mass (kg)","Muscle mass (kg)","Hydration (kg)"
    ])
    other_rows=[]
    prior_dt="2026-08-28 09:30:00"
    prior_vals={"Left Arm":(4.4,.8),"Right Arm":(4.5,.8),"Torso":(35.2,13.7),"Left leg":(11.8,3.0),"Right leg":(11.8,3.0)}
    for pos,(m,f) in prior_vals.items():
        other_rows += [
            ["Muscle Mass for segments",prior_dt,m,"kg",pos],
            ["Fat Mass for segments in mass unit",prior_dt,f,"kg",pos],
        ]
    dt="2026-09-20 10:10:00"
    vals={"Left Arm":(4.6,.7),"Right Arm":(4.8,.7),"Torso":(36.7,12.2),"Left leg":(12.0,2.6),"Right leg":(11.8,2.9)}
    for pos,(m,f) in vals.items():
        other_rows += [
            ["Muscle Mass for segments",dt,m,"kg",pos],
            ["Fat Mass for segments in mass unit",dt,f,"kg",pos],
        ]
    d2="2026-09-22 10:26:00"
    other_rows += [
        ["Visceral fat",d2,3.7,"", ""],
        ["BMR",d2,2145,"kcal/day", ""],
        ["Metabolic age",d2,42,"year", ""],
        ["Vascular age",d2,46,"year", ""],
        ["ICW",d2,31,"kg", ""],
        ["ECW",d2,19,"kg", ""],
    ]
    other=pd.DataFrame(other_rows,columns=["type","date","value","unit","position"])
    return normalize_frames(weight, other, {})


def test_core_numbers_and_silhouette():
    w,o,meta=synthetic_frames()
    p=Profile("Mustafa","Male",1.82,date(1982,10,20),None,82.5)
    r=build_report_data(w,o,p,diagnostics=meta)
    assert abs(r.weight_kg-91.94)<1e-9  # headline always latest complete scan
    assert round(r.body_fat_pct,1)==20.0
    assert round(r.muscle_pct,1)==76.1
    assert r.segment_date.strftime('%Y-%m-%d')=='2026-09-20'
    assert round(r.segments['Left Arm'].muscle_kg,1)==4.6
    assert r.diagnostics['previous_segment_date'].strftime('%Y-%m-%d')=='2026-08-28'
    html=render_report_html(r)
    assert "91.94 kg" in html
    assert "20.0%" in html
    assert "69.98 kg" in html
    assert "82.5 kg" in html
    assert "regional body silhouette" in html


def test_earliest_daily_rule_is_default():
    w,o,meta=synthetic_frames()
    p=Profile("Mustafa","Male",1.82,date(1982,10,20),None,82.5)
    r=build_report_data(w,o,p,diagnostics=meta)
    sep22=r.history[pd.to_datetime(r.history["Date"]).dt.date==date(2026,9,22)]
    assert len(sep22)==1
    assert sep22.iloc[0]["Weight (kg)"]==92.30


def test_latest_daily_rule():
    w,o,meta=synthetic_frames()
    p=Profile("Mustafa","Male",1.82,date(1982,10,20),None,82.5,"Latest complete scan",8)
    r=build_report_data(w,o,p,diagnostics=meta)
    sep22=r.history[pd.to_datetime(r.history["Date"]).dt.date==date(2026,9,22)]
    assert len(sep22)==1
    assert sep22.iloc[0]["Weight (kg)"]==91.94


def test_daily_median_rule_and_point_count():
    w,o,meta=synthetic_frames()
    p=Profile("Mustafa","Male",1.82,date(1982,10,20),None,82.5,"Daily median",6)
    r=build_report_data(w,o,p,diagnostics=meta)
    sep22=r.history[pd.to_datetime(r.history["Date"]).dt.date==date(2026,9,22)]
    assert len(sep22)==1
    assert round(float(sep22.iloc[0]["Weight (kg)"]),2)==92.12
    assert len(r.history)<=6
    assert r.diagnostics["history_daily_rule"]=="Daily median"
    assert r.diagnostics["history_points"]==6



def test_v6_practitioner_identity_and_change_summary():
    w,o,meta=synthetic_frames()
    p=Profile(
        "Mustafa Mansour","Male",1.82,date(1982,10,20),None,82.0,
        "Earliest complete scan",8,"MUS-001"
    )
    r=build_report_data(w,o,p,diagnostics=meta)
    assert r.diagnostics["previous_scan_date"].strftime("%Y-%m-%d")=="2026-09-21"
    assert r.diagnostics["previous_weight_kg"]==92.6
    assert r.diagnostics["median30_scan_count"]==7
    html=render_report_html(r)
    assert "Mustafa Mansour" in html
    assert "ID MUS-001" in html
    assert "TREND & DATA QUALITY" in html
    assert "Previous measured day" in html
    assert "9.94 kg to goal" in html


def test_previous_day_respects_selected_history_rule():
    w,o,meta=synthetic_frames()
    # Latest daily rule still compares the current headline scan against the
    # representative reading from the previous calendar day, not a same-day repeat.
    p=Profile("Mustafa","Male",1.82,date(1982,10,20),None,82.0,"Latest complete scan",8)
    r=build_report_data(w,o,p,diagnostics=meta)
    assert r.diagnostics["previous_scan_date"].strftime("%Y-%m-%d")=="2026-09-21"
    assert r.diagnostics["previous_weight_kg"]==92.6


def test_v62_segmental_change_and_torso_chip():
    w,o,meta=synthetic_frames()
    p=Profile("Mustafa","Male",1.82,date(1982,10,20),None,82.0,"Earliest complete scan",8)
    r=build_report_data(w,o,p,diagnostics=meta)
    html=render_report_html(r)
    assert "seg-torso-chip" in html
    assert "Largest observed muscle change" in html
    assert "Largest observed fat change" in html
    assert "Torso ↑ 1.5 kg" in html
    assert "Torso ↓ 1.5 kg" in html
    assert "vs 28 Aug 2026" in html


def test_v62_history_responsive_and_pp_kept():
    w,o,meta=synthetic_frames()
    p=Profile("Mustafa","Male",1.82,date(1982,10,20),None,82.0,"Earliest complete scan",12)
    r=build_report_data(w,o,p,diagnostics=meta)
    html=render_report_html(r)
    assert "viewBox='0 0 1000 78'" in html
    assert "hlatest-halo" in html
    assert "pp" in html
    assert "12-day" not in html or "-DAY Δ" in html

if __name__ == '__main__':
    test_core_numbers_and_silhouette()
    test_earliest_daily_rule_is_default()
    test_latest_daily_rule()
    test_daily_median_rule_and_point_count()
    test_v6_practitioner_identity_and_change_summary()
    test_previous_day_respects_selected_history_rule()
    test_v62_segmental_change_and_torso_chip()
    test_v62_history_responsive_and_pp_kept()
    print('OK')


def extended_frames():
    rows=[]
    start=pd.Timestamp("2026-06-15 07:30:00")
    # approximately twice weekly plus dense final week, with gradual fat loss and muscle retention
    dates=list(pd.date_range(start, pd.Timestamp("2026-09-10 07:30:00"), freq="7D"))
    dates += list(pd.date_range(pd.Timestamp("2026-09-11 07:20:00"), pd.Timestamp("2026-09-22 07:20:00"), freq="1D"))
    dates=sorted(set(dates))
    for i,dt in enumerate(dates):
        progress=i/max(1,len(dates)-1)
        weight=95.0-3.0*progress
        fat_pct=23.0-2.6*progress
        fat=weight*fat_pct/100
        muscle=69.0+0.7*progress
        bone=3.5
        water=weight*0.535
        rows.append([dt,weight,fat,bone,muscle,water])
    # headline later same day
    rows.append([pd.Timestamp("2026-09-22 10:26:00"),91.94,18.43,3.53,69.98,49.28])
    weight=pd.DataFrame(rows,columns=["Date","Weight (kg)","Fat mass (kg)","Bone mass (kg)","Muscle mass (kg)","Hydration (kg)"])
    other_rows=[]
    for dt,vals in [
        ("2026-08-28 09:30:00",{"Left Arm":(4.4,.8),"Right Arm":(4.5,.8),"Torso":(35.2,13.7),"Left leg":(11.8,3.0),"Right leg":(11.8,3.0)}),
        ("2026-09-20 10:10:00",{"Left Arm":(4.6,.7),"Right Arm":(4.8,.7),"Torso":(36.7,12.2),"Left leg":(12.0,2.6),"Right leg":(11.8,2.9)}),
    ]:
        for pos,(m,f) in vals.items():
            other_rows += [["Muscle Mass for segments",dt,m,"kg",pos],["Fat Mass for segments in mass unit",dt,f,"kg",pos]]
    other_rows += [["Visceral fat","2026-09-22 10:26:00",3.7,"",""],["Metabolic age","2026-09-22 10:26:00",42,"year",""],["Vascular age","2026-09-22 10:26:00",46,"year",""],["ICW","2026-09-22 10:26:00",31,"kg",""],["ECW","2026-09-22 10:26:00",19,"kg",""]]
    other=pd.DataFrame(other_rows,columns=["type","date","value","unit","position"])
    return normalize_frames(weight,other,{})


def test_v7_longitudinal_trends_and_consistency():
    w,o,meta=extended_frames()
    p=Profile("Mustafa","Male",1.82,date(1982,10,20),None,82.0,"Earliest complete scan",12,None,date(2026,6,15),95.0,15.0)
    r=build_report_data(w,o,p,diagnostics=meta)
    assert r.diagnostics.get("trend30")
    assert r.diagnostics.get("trend90")
    assert r.diagnostics["trend30"]["span_days"] >= 21
    assert r.diagnostics["trend90"]["span_days"] >= 60
    assert "measurement_consistency" in r.diagnostics
    html=render_report_html(r)
    assert "30-DAY TREND" in html and "90-DAY TREND" in html
    assert "Measurement timing" in html
    assert "% complete" in html
    assert "Body-fat goal" in html
    assert "Withings Body Scan · practitioner-friendly" not in html
    assert "<div class='subtitle'>Withings Body Scan</div>" in html


def test_v7_hides_unavailable_bmr_and_assessment_has_values():
    w,o,meta=extended_frames()
    p=Profile("Mustafa","Male",1.82,date(1982,10,20),None,82.0)
    r=build_report_data(w,o,p,diagnostics=meta)
    assert r.bmr is None
    html=render_report_html(r)
    # BMR should not appear as an unavailable parameter row.
    assert "<span>BMR</span><b>-</b>" not in html
    assert "assess-value" in html
    assert f"{r.body_fat_pct:.1f}%" in html
