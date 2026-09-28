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
    assert "CHANGE SUMMARY" in html
    assert "Since previous measured day" in html
    assert "9.94 kg to goal" in html


def test_previous_day_respects_selected_history_rule():
    w,o,meta=synthetic_frames()
    # Latest daily rule still compares the current headline scan against the
    # representative reading from the previous calendar day, not a same-day repeat.
    p=Profile("Mustafa","Male",1.82,date(1982,10,20),None,82.0,"Latest complete scan",8)
    r=build_report_data(w,o,p,diagnostics=meta)
    assert r.diagnostics["previous_scan_date"].strftime("%Y-%m-%d")=="2026-09-21"
    assert r.diagnostics["previous_weight_kg"]==92.6

if __name__ == '__main__':
    test_core_numbers_and_silhouette()
    test_earliest_daily_rule_is_default()
    test_latest_daily_rule()
    test_daily_median_rule_and_point_count()
    test_v6_practitioner_identity_and_change_summary()
    test_previous_day_respects_selected_history_rule()
    print('OK')
