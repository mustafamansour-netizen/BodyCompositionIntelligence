from datetime import date
import io, zipfile
import pandas as pd

from report_engine import Profile, build_report_data, load_withings_zip, render_report_html


def synthetic_frames():
    rows = [
        ["2026-09-17 10:00:00",92.2,19.73,3.5,69.0,49.0],
        ["2026-09-18 10:00:00",92.5,18.78,3.5,70.1,49.2],
        ["2026-09-19 10:00:00",92.4,19.22,3.5,69.6,49.1],
        ["2026-09-20 10:00:00",92.6,20.37,3.5,68.8,49.0],
        ["2026-09-21 10:00:00",92.6,20.09,3.5,69.0,49.1],
        ["2026-09-22 10:26:00",91.94,18.43,3.53,69.98,49.28],
    ]
    weight = pd.DataFrame(rows, columns=["Date","Weight (kg)","Fat mass (kg)","Bone mass (kg)","Muscle mass (kg)","Hydration (kg)"])
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
    return weight,other


def test_core_numbers():
    w,o=synthetic_frames()
    from report_engine import normalize_frames
    w,o,meta=normalize_frames(w,o,{})
    p=Profile("Mustafa","Male",1.82,date(1982,10,20),None,82.0)
    r=build_report_data(w,o,p,diagnostics=meta)
    assert abs(r.weight_kg-91.94)<1e-9
    assert round(r.body_fat_pct,1)==20.0
    assert round(r.muscle_pct,1)==76.1
    assert r.segment_date.strftime('%Y-%m-%d')=='2026-09-20'
    assert round(r.segments['Left Arm'].muscle_kg,1)==4.6
    html=render_report_html(r)
    assert "91.94 kg" in html
    assert "20.0%" in html
    assert "69.98 kg" in html
    assert "82.0 kg" in html


if __name__ == '__main__':
    test_core_numbers()
    print('OK')
