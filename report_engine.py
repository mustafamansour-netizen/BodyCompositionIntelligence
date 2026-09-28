from __future__ import annotations

import io
import math
import re
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime
from html import escape
from pathlib import PurePosixPath
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Data model
# -----------------------------

@dataclass
class Profile:
    name: str
    sex: str  # Male / Female
    height_m: float
    birth_date: Optional[date] = None
    age_override: Optional[int] = None
    goal_weight_kg: Optional[float] = None


@dataclass
class ReferenceBands:
    body_fat: Tuple[float, float]
    muscle_pct: Tuple[float, float]
    water_pct: Tuple[float, float]
    bone_pct: Tuple[float, float]
    visceral_fat: Tuple[float, float] = (0.0, 5.0)
    bmi: Tuple[float, float] = (18.5, 24.9)


@dataclass
class SegmentValue:
    muscle_kg: Optional[float] = None
    fat_kg: Optional[float] = None
    ffm_kg: Optional[float] = None


@dataclass
class ReportData:
    profile: Profile
    bands: ReferenceBands
    scan_date: pd.Timestamp
    segment_date: Optional[pd.Timestamp]
    weight_kg: float
    fat_mass_kg: float
    muscle_mass_kg: float
    bone_mass_kg: float
    water_kg: float
    visceral_fat: Optional[float]
    bmr: Optional[float]
    metabolic_age: Optional[float]
    vascular_age: Optional[float]
    icw_kg: Optional[float]
    ecw_kg: Optional[float]
    segments: Dict[str, SegmentValue]
    history: pd.DataFrame
    monthly: pd.DataFrame
    diagnostics: Dict[str, object] = field(default_factory=dict)

    @property
    def fat_free_mass_kg(self) -> float:
        return self.weight_kg - self.fat_mass_kg

    @property
    def body_fat_pct(self) -> float:
        return self.fat_mass_kg / self.weight_kg * 100

    @property
    def muscle_pct(self) -> float:
        return self.muscle_mass_kg / self.weight_kg * 100

    @property
    def water_pct(self) -> float:
        return self.water_kg / self.weight_kg * 100

    @property
    def bone_pct(self) -> float:
        return self.bone_mass_kg / self.weight_kg * 100

    @property
    def bmi(self) -> float:
        return self.weight_kg / (self.profile.height_m ** 2)

    @property
    def ecw_tbw_pct(self) -> Optional[float]:
        if self.icw_kg is None or self.ecw_kg is None:
            return None
        denom = self.icw_kg + self.ecw_kg
        if denom <= 0:
            return None
        return self.ecw_kg / denom * 100


# -----------------------------
# Withings import
# -----------------------------

WEIGHT_REQUIRED = {
    "Date",
    "Weight (kg)",
    "Fat mass (kg)",
    "Bone mass (kg)",
    "Muscle mass (kg)",
    "Hydration (kg)",
}

OTHER_REQUIRED = {"type", "date", "value", "unit", "position"}


def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    last_err = None
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "cp1252", "latin1"):
        try:
            text = raw.decode(encoding)
            # Withings exports are comma-separated, but sniff semicolon if needed.
            first = text.splitlines()[0] if text.splitlines() else ""
            sep = ";" if first.count(";") > first.count(",") else ","
            return pd.read_csv(io.StringIO(text), sep=sep)
        except Exception as exc:
            last_err = exc
    raise ValueError(f"Could not read CSV: {last_err}")


def _find_member(zf: zipfile.ZipFile, basename: str) -> Optional[str]:
    basename = basename.lower()
    candidates = []
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        if PurePosixPath(name).name.lower() == basename:
            candidates.append(name)
    if not candidates:
        return None
    # Prefer the shortest path (normally the canonical export file).
    return sorted(candidates, key=lambda x: (x.count("/"), len(x)))[0]


def load_withings_zip(raw_zip: bytes) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    with zipfile.ZipFile(io.BytesIO(raw_zip), "r") as zf:
        weight_name = _find_member(zf, "weight.csv")
        other_name = _find_member(zf, "other.csv")
        if weight_name is None or other_name is None:
            found = [PurePosixPath(x).name for x in zf.namelist() if not x.endswith("/")]
            raise ValueError(
                "The ZIP must contain weight.csv and other.csv. "
                f"Found {len(found)} files but one/both required files were missing."
            )
        weight_df = _read_csv_bytes(zf.read(weight_name))
        other_df = _read_csv_bytes(zf.read(other_name))
        meta = {"weight_member": weight_name, "other_member": other_name}
    return normalize_frames(weight_df, other_df, meta)


def load_withings_csvs(weight_raw: bytes, other_raw: bytes) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    weight_df = _read_csv_bytes(weight_raw)
    other_df = _read_csv_bytes(other_raw)
    return normalize_frames(weight_df, other_df, {"weight_member": "weight.csv", "other_member": "other.csv"})


def normalize_frames(weight_df: pd.DataFrame, other_df: pd.DataFrame, meta=None):
    meta = dict(meta or {})
    missing_w = sorted(WEIGHT_REQUIRED - set(weight_df.columns))
    missing_o = sorted(OTHER_REQUIRED - set(other_df.columns))
    if missing_w:
        raise ValueError("Missing weight.csv columns: " + ", ".join(missing_w))
    if missing_o:
        raise ValueError("Missing other.csv columns: " + ", ".join(missing_o))

    weight_df = weight_df.copy()
    other_df = other_df.copy()

    weight_df["Date"] = pd.to_datetime(weight_df["Date"], errors="coerce")
    # Remove timezone to simplify grouping/display while preserving wall-clock export time.
    try:
        if getattr(weight_df["Date"].dt, "tz", None) is not None:
            weight_df["Date"] = weight_df["Date"].dt.tz_localize(None)
    except Exception:
        pass

    for col in ["Weight (kg)", "Fat mass (kg)", "Bone mass (kg)", "Muscle mass (kg)", "Hydration (kg)"]:
        weight_df[col] = pd.to_numeric(weight_df[col], errors="coerce")

    other_df["date"] = pd.to_datetime(other_df["date"], errors="coerce")
    try:
        if getattr(other_df["date"].dt, "tz", None) is not None:
            other_df["date"] = other_df["date"].dt.tz_localize(None)
    except Exception:
        pass
    other_df["value_numeric"] = pd.to_numeric(other_df["value"], errors="coerce")

    weight_df = weight_df.dropna(subset=["Date", "Weight (kg)"]).sort_values("Date").reset_index(drop=True)
    other_df = other_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    meta.update(
        {
            "weight_records": len(weight_df),
            "other_records": len(other_df),
            "other_types": sorted(other_df["type"].dropna().astype(str).unique().tolist()),
            "positions": sorted(other_df["position"].dropna().astype(str).unique().tolist()),
        }
    )
    return weight_df, other_df, meta


# -----------------------------
# Matching and calculations
# -----------------------------


def age_on(profile: Profile, when: pd.Timestamp) -> Optional[int]:
    if profile.birth_date:
        d = when.date()
        b = profile.birth_date
        return d.year - b.year - ((d.month, d.day) < (b.month, b.day))
    return profile.age_override


def default_bands(sex: str) -> ReferenceBands:
    # These are the reference bands used in the approved report template for the
    # user's current age group. They are exposed in the UI so they can be changed.
    if str(sex).lower().startswith("f"):
        return ReferenceBands(
            body_fat=(24.0, 34.0),
            muscle_pct=(62.0, 73.5),
            water_pct=(45.0, 60.0),
            bone_pct=(2.5, 4.0),
        )
    return ReferenceBands(
        body_fat=(11.0, 22.0),
        muscle_pct=(73.0, 86.0),
        water_pct=(50.0, 65.0),
        bone_pct=(3.0, 5.0),
    )


def _norm(s: object) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _canonical_position(value: object) -> Optional[str]:
    s = _norm(value)
    mapping = {
        "left arm": "Left Arm",
        "right arm": "Right Arm",
        "left leg": "Left Leg",
        "right leg": "Right Leg",
        "torso": "Torso",
        "trunk": "Torso",
    }
    if s in mapping:
        return mapping[s]
    if "left" in s and "arm" in s:
        return "Left Arm"
    if "right" in s and "arm" in s:
        return "Right Arm"
    if "left" in s and "leg" in s:
        return "Left Leg"
    if "right" in s and "leg" in s:
        return "Right Leg"
    if "torso" in s or "trunk" in s:
        return "Torso"
    return None


def _metric_kind(type_value: object) -> Optional[str]:
    s = _norm(type_value)
    # Segment metrics first.
    if "segment" in s and "muscle" in s and "mass" in s:
        return "segment_muscle"
    if "segment" in s and "fat free" in s:
        return "segment_ffm"
    if "segment" in s and "fat" in s and "mass" in s:
        return "segment_fat"

    if "visceral" in s and "fat" in s:
        return "visceral_fat"
    if s == "bmr" or "basal metabolic" in s:
        return "bmr"
    if "metabolic" in s and "age" in s:
        return "metabolic_age"
    if "vascular" in s and "age" in s:
        return "vascular_age"
    if s == "icw" or "intracellular water" in s or "intra cellular water" in s:
        return "icw"
    if s == "ecw" or "extracellular water" in s or "extra cellular water" in s:
        return "ecw"
    return None


def _latest_metric(other_df: pd.DataFrame, kind: str, at_or_before: pd.Timestamp) -> Optional[float]:
    if other_df.empty:
        return None
    mask = other_df["type"].map(_metric_kind).eq(kind) & other_df["value_numeric"].notna()
    subset = other_df.loc[mask].copy()
    if subset.empty:
        return None
    # Avoid leaking a later scan into an earlier report. Allow a modest same-day timestamp drift.
    subset = subset[subset["date"] <= at_or_before + pd.Timedelta(hours=18)]
    if subset.empty:
        return None
    subset["delta"] = (subset["date"] - at_or_before).abs()
    row = subset.sort_values(["date", "delta"]).iloc[-1]
    return float(row["value_numeric"])


def _find_latest_segment_snapshot(other_df: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Dict[str, SegmentValue], Dict[str, object]]:
    seg = other_df.copy()
    seg["kind"] = seg["type"].map(_metric_kind)
    seg["position_canon"] = seg["position"].map(_canonical_position)
    seg = seg[
        seg["kind"].isin(["segment_muscle", "segment_fat", "segment_ffm"])
        & seg["position_canon"].notna()
        & seg["value_numeric"].notna()
    ].copy()

    empty = {p: SegmentValue() for p in ["Left Arm", "Right Arm", "Torso", "Left Leg", "Right Leg"]}
    if seg.empty:
        return None, empty, {"segment_candidates": 0}

    seg["scan_key"] = seg["date"].dt.floor("min")
    target_positions = {"Left Arm", "Right Arm", "Torso", "Left Leg", "Right Leg"}

    candidates = []
    for scan_key, g in seg.groupby("scan_key"):
        mus_pos = set(g.loc[g["kind"] == "segment_muscle", "position_canon"])
        fat_pos = set(g.loc[g["kind"] == "segment_fat", "position_canon"])
        complete = len(target_positions & mus_pos & fat_pos)
        candidates.append((scan_key, complete, len(g)))

    # Prefer the latest fully complete snapshot; otherwise latest best-coverage snapshot.
    full = [x for x in candidates if x[1] == 5]
    if full:
        scan_key = max(full, key=lambda x: x[0])[0]
    else:
        best_coverage = max(x[1] for x in candidates)
        scan_key = max([x for x in candidates if x[1] == best_coverage], key=lambda x: x[0])[0]

    g = seg[seg["scan_key"] == scan_key]
    out: Dict[str, SegmentValue] = {}
    for pos in ["Left Arm", "Right Arm", "Torso", "Left Leg", "Right Leg"]:
        pg = g[g["position_canon"] == pos]
        vals = {}
        for kind, field_name in [
            ("segment_muscle", "muscle_kg"),
            ("segment_fat", "fat_kg"),
            ("segment_ffm", "ffm_kg"),
        ]:
            kg = pg.loc[pg["kind"] == kind, "value_numeric"]
            vals[field_name] = float(kg.iloc[-1]) if not kg.empty else None
        out[pos] = SegmentValue(**vals)

    return pd.Timestamp(scan_key), out, {
        "segment_candidates": len(candidates),
        "segment_complete_positions": next((x[1] for x in candidates if x[0] == scan_key), 0),
    }


def _monthly_history(complete: pd.DataFrame, latest_date: pd.Timestamp) -> pd.DataFrame:
    start = (latest_date.to_period("M") - 11).to_timestamp()
    end = latest_date.to_period("M").to_timestamp()
    months = pd.date_range(start, end, freq="MS")

    df = complete.copy()
    df["Body fat %"] = df["Fat mass (kg)"] / df["Weight (kg)"] * 100
    df["month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    med = (
        df[df["month"].between(start, end)]
        .groupby("month")[["Weight (kg)", "Body fat %", "Muscle mass (kg)"]]
        .median()
        .reindex(months)
    )
    med.index.name = "Month"
    return med.reset_index()


def build_report_data(
    weight_df: pd.DataFrame,
    other_df: pd.DataFrame,
    profile: Profile,
    bands: Optional[ReferenceBands] = None,
    diagnostics: Optional[Dict[str, object]] = None,
) -> ReportData:
    complete = weight_df.dropna(
        subset=["Date", "Weight (kg)", "Fat mass (kg)", "Bone mass (kg)", "Muscle mass (kg)", "Hydration (kg)"]
    ).sort_values("Date")
    if complete.empty:
        raise ValueError("No complete Withings body-composition scan was found in weight.csv.")

    latest = complete.iloc[-1]
    scan_date = pd.Timestamp(latest["Date"])
    bands = bands or default_bands(profile.sex)

    segment_date, segments, seg_diag = _find_latest_segment_snapshot(other_df)
    history = complete.tail(6).copy().sort_values("Date", ascending=False)
    history["Body fat %"] = history["Fat mass (kg)"] / history["Weight (kg)"] * 100
    monthly = _monthly_history(complete, scan_date)

    diag = dict(diagnostics or {})
    diag.update(seg_diag)
    diag["complete_scans"] = int(len(complete))
    diag["latest_complete_scan"] = scan_date

    return ReportData(
        profile=profile,
        bands=bands,
        scan_date=scan_date,
        segment_date=segment_date,
        weight_kg=float(latest["Weight (kg)"]),
        fat_mass_kg=float(latest["Fat mass (kg)"]),
        muscle_mass_kg=float(latest["Muscle mass (kg)"]),
        bone_mass_kg=float(latest["Bone mass (kg)"]),
        water_kg=float(latest["Hydration (kg)"]),
        visceral_fat=_latest_metric(other_df, "visceral_fat", scan_date),
        bmr=_latest_metric(other_df, "bmr", scan_date),
        metabolic_age=_latest_metric(other_df, "metabolic_age", scan_date),
        vascular_age=_latest_metric(other_df, "vascular_age", scan_date),
        icw_kg=_latest_metric(other_df, "icw", scan_date),
        ecw_kg=_latest_metric(other_df, "ecw", scan_date),
        segments=segments,
        history=history,
        monthly=monthly,
        diagnostics=diag,
    )


# -----------------------------
# Render helpers
# -----------------------------


def fmt_num(v: Optional[float], decimals=1, suffix="") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.{decimals}f}{suffix}"


def _status(value: Optional[float], band: Tuple[float, float], *, just_margin: float = 0.0) -> Tuple[str, str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "No data", "neutral"
    lo, hi = band
    if lo <= value <= hi:
        return "Within ref.", "good"
    if value > hi:
        if just_margin and value <= hi + just_margin:
            return "Just above", "warn"
        return "Above ref.", "bad" if value > hi + max(just_margin, 2.0) else "warn"
    if just_margin and value >= lo - just_margin:
        return "Just below", "warn"
    return "Below ref.", "warn"


def _visceral_status(v: Optional[float], band=(0.0, 5.0)) -> Tuple[str, str]:
    if v is None:
        return "No data", "neutral"
    if band[0] <= v <= band[1]:
        return "Normal", "good"
    return "Elevated", "bad"


def _bmi_status(v: float, band=(18.5, 24.9)) -> Tuple[str, str]:
    lo, hi = band
    if lo <= v <= hi:
        return "Within ref.", "good"
    if hi < v <= hi + 0.7:
        return "Just above", "warn"
    if v > hi:
        return "Above normal", "warn"
    if lo - 0.7 <= v < lo:
        return "Just below", "warn"
    return "Below normal", "warn"


def _dot_class(status_class: str) -> str:
    return status_class if status_class in {"good", "warn", "bad", "neutral"} else "neutral"


def _bar(value: float, ref: Tuple[float, float], domain: Tuple[float, float], label: str) -> str:
    dlo, dhi = domain
    rlo, rhi = ref
    def pct(x):
        return max(0.0, min(100.0, (x - dlo) / (dhi - dlo) * 100.0))
    rp1, rp2, vp = pct(rlo), pct(rhi), pct(value)
    return f"""
    <div class='bar-row'>
      <div class='bar-label'>{escape(label)}</div>
      <div class='bar-mid'>
        <div class='bar-note'>{fmt_num(rlo,1)}-{fmt_num(rhi,1)}{('%' if '%' in label else '')}</div>
        <div class='track'><div class='ref-zone' style='left:{rp1:.1f}%;width:{max(1,rp2-rp1):.1f}%'></div><span class='marker' style='left:{vp:.1f}%'></span></div>
      </div>
      <div class='bar-value'>{fmt_num(value,1)}{('%' if '%' in label else (' kg' if label=='Weight' else ''))}</div>
    </div>"""


def _balance(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    avg = (a + b) / 2
    return abs(a - b) / avg * 100 if avg else None


def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _mix(c1: str, c2: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    a, b = _hex_to_rgb(c1), _hex_to_rgb(c2)
    vals = [round(a[i] + (b[i] - a[i]) * t) for i in range(3)]
    return "#%02x%02x%02x" % tuple(vals)


def _segment_metric_value(seg: SegmentValue, metric: str) -> Optional[float]:
    return seg.muscle_kg if metric == "muscle" else seg.fat_kg


def _segment_metric_colors(segments: Dict[str, SegmentValue], metric: str) -> Dict[str, str]:
    base = "#1688a6" if metric == "muscle" else "#d79a45"
    values = {
        pos: _segment_metric_value(seg, metric)
        for pos, seg in segments.items()
        if _segment_metric_value(seg, metric) is not None
    }
    finite = [float(v) for v in values.values() if v is not None and float(v) >= 0]
    if not finite:
        return {p: "#dfe7eb" for p in segments}
    vmax = max(finite) or 1.0
    out = {}
    for pos in segments:
        v = values.get(pos)
        if v is None:
            out[pos] = "#e6ecef"
        else:
            # Intensity represents only regional share within this scan. It is not a reference range.
            t = max(0.0, min(1.0, float(v) / vmax))
            out[pos] = _mix("#edf2f4", base, 0.26 + 0.70 * (t ** 0.72))
    return out


def _segment_body_svg(data: ReportData, metric: str) -> str:
    c = _segment_metric_colors(data.segments, metric)
    la, ra, torso, ll, rl = [c.get(x, "#dfe7eb") for x in ["Left Arm", "Right Arm", "Torso", "Left Leg", "Right Leg"]]
    female = str(data.profile.sex).lower().startswith("f")
    if female:
        torso_path = "M84 58 C90 50 99 47 110 47 C121 47 130 50 136 58 C139 72 138 88 133 104 C130 114 132 126 141 142 C132 150 121 154 110 154 C99 154 88 150 79 142 C88 126 90 114 87 104 C82 88 81 72 84 58 Z"
        pelvis_path = "M80 141 C89 149 99 153 110 153 C121 153 131 149 140 141 L136 165 C127 171 120 174 110 174 C100 174 93 171 84 165 Z"
    else:
        torso_path = "M80 58 C88 49 98 46 110 46 C122 46 132 49 140 58 C144 75 143 94 138 111 C134 127 124 141 110 148 C96 141 86 127 82 111 C77 94 76 75 80 58 Z"
        pelvis_path = "M83 142 C92 148 101 151 110 151 C119 151 128 148 137 142 L135 166 C126 171 118 174 110 174 C102 174 94 171 85 166 Z"
    # Smooth, neutral medical-style silhouette with independently shaded regions.
    return f"""
    <svg viewBox='0 0 220 320' class='segment-body-svg' role='img' aria-label='{metric.title()} regional distribution body schematic'>
      <defs>
        <linearGradient id='head-{metric}' x1='0' y1='0' x2='0' y2='1'><stop offset='0' stop-color='#edf1f3'/><stop offset='1' stop-color='#cfd8dd'/></linearGradient>
        <filter id='shadow-{metric}' x='-20%' y='-20%' width='140%' height='140%'><feDropShadow dx='0' dy='1.4' stdDeviation='2.0' flood-color='#6e8795' flood-opacity='.16'/></filter>
      </defs>
      <g filter='url(#shadow-{metric})' stroke='#cbd6dc' stroke-width='1.15' stroke-linejoin='round'>
        <ellipse cx='110' cy='27' rx='17' ry='20' fill='url(#head-{metric})'/>
        <path d='M102 45 C104 51 104 54 102 59 L118 59 C116 54 116 51 118 45 Z' fill='#dde5e9'/>
        <path d='{torso_path}' fill='{torso}'/>
        <path d='M82 63 C68 67 60 77 55 91 L34 151 C30 162 34 169 41 168 C47 168 50 163 53 155 L73 111 C78 101 84 93 88 86 Z' fill='{la}'/>
        <path d='M138 63 C152 67 160 77 165 91 L186 151 C190 162 186 169 179 168 C173 168 170 163 167 155 L147 111 C142 101 136 93 132 86 Z' fill='{ra}'/>
        <path d='{pelvis_path}' fill='#dfe6e9'/>
        <path d='M87 165 C80 190 79 216 82 248 L78 294 C77 305 82 311 90 310 L100 306 L103 248 L108 174 Z' fill='{ll}'/>
        <path d='M133 165 C140 190 141 216 138 248 L142 294 C143 305 138 311 130 310 L120 306 L117 248 L112 174 Z' fill='{rl}'/>
        <path d='M78 293 C74 302 68 307 61 311 L91 311 L91 304 Z' fill='#dce4e8'/>
        <path d='M142 293 C146 302 152 307 159 311 L129 311 L129 304 Z' fill='#dce4e8'/>
      </g>
    </svg>"""


def _segment_label(position: str, value: Optional[float], align: str = "left") -> str:
    return f"""<div class='map-label {align}'><span>{escape(position.upper())}</span><b>{fmt_num(value,1,' kg')}</b></div>"""


def _segment_map(data: ReportData, metric: str) -> str:
    seg = data.segments
    la, ra = seg.get("Left Arm", SegmentValue()), seg.get("Right Arm", SegmentValue())
    ll, rl = seg.get("Left Leg", SegmentValue()), seg.get("Right Leg", SegmentValue())
    torso = seg.get("Torso", SegmentValue())
    getv = lambda x: _segment_metric_value(x, metric)
    title = "MUSCLE DISTRIBUTION" if metric == "muscle" else "FAT DISTRIBUTION"
    accent = "muscle" if metric == "muscle" else "fat"
    return f"""
      <div class='map-panel {accent}'>
        <div class='map-head'><b>{title}</b><span>regional mass</span></div>
        <div class='map-content'>
          <div class='map-side left-side'>
            {_segment_label('Left Arm', getv(la), 'right')}
            {_segment_label('Left Leg', getv(ll), 'right')}
          </div>
          <div class='map-center'>
            <div class='torso-chip'><span>TORSO</span><b>{fmt_num(getv(torso),1,' kg')}</b></div>
            {_segment_body_svg(data, metric)}
          </div>
          <div class='map-side right-side'>
            {_segment_label('Right Arm', getv(ra), 'left')}
            {_segment_label('Right Leg', getv(rl), 'left')}
          </div>
        </div>
      </div>"""


def _balance_card(title: str, left: Optional[float], right: Optional[float]) -> str:
    diff = _balance(left, right)
    if left is None or right is None:
        return f"<div class='balance-card'><div class='balance-title'>{escape(title)}</div><div class='balance-missing'>No paired muscle values</div></div>"
    mx = max(left, right, 0.0001)
    lw = left / mx * 100
    rw = right / mx * 100
    return f"""
      <div class='balance-card'>
        <div class='balance-head'><b>{escape(title)}</b><span>{fmt_num(diff,1,'%')} difference</span></div>
        <div class='balance-values'><span>L&nbsp; <b>{left:.1f} kg</b></span><span>R&nbsp; <b>{right:.1f} kg</b></span></div>
        <div class='balance-track'><div class='balance-half left'><i style='width:{lw:.1f}%'></i></div><em></em><div class='balance-half right'><i style='width:{rw:.1f}%'></i></div></div>
      </div>"""


def _sparkline_svg(values: List[Optional[float]], width=250, height=58) -> str:
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return f"<svg viewBox='0 0 {width} {height}' class='spark'></svg>"
    ymin, ymax = float(finite.min()), float(finite.max())
    if math.isclose(ymin, ymax):
        ymin -= 1.0
        ymax += 1.0
    pad_y = (ymax - ymin) * 0.18
    ymin -= pad_y
    ymax += pad_y
    n = len(arr)
    xs = np.linspace(7, width - 7, n) if n > 1 else np.array([width / 2])
    def y(v):
        return height - 7 - (v - ymin) / (ymax - ymin) * (height - 14)

    segments = []
    current = []
    circles = []
    for x, v in zip(xs, arr):
        if np.isfinite(v):
            current.append((x, y(v)))
            circles.append(f"<circle cx='{x:.1f}' cy='{y(v):.1f}' r='2.4' fill='#1182ba'/>")
        else:
            if len(current) >= 2:
                segments.append(current)
            current = []
    if len(current) >= 2:
        segments.append(current)

    paths = []
    for seg in segments:
        pts = " ".join(f"{x:.1f},{yy:.1f}" for x, yy in seg)
        paths.append(f"<polyline points='{pts}' fill='none' stroke='#1182ba' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/>")

    return f"<svg viewBox='0 0 {width} {height}' class='spark'><line x1='6' x2='{width-6}' y1='{height-7}' y2='{height-7}' stroke='#e7edf1' stroke-width='1'/>{''.join(paths)}{''.join(circles)}</svg>"


def _history_chart(monthly: pd.DataFrame, col: str, title: str, suffix: str, decimals=1) -> str:
    vals = [None if pd.isna(v) else float(v) for v in monthly[col].tolist()]
    finite = [v for v in vals if v is not None]
    latest = finite[-1] if finite else None
    first_label = monthly["Month"].iloc[0].strftime("%b %y") if len(monthly) else ""
    last_label = monthly["Month"].iloc[-1].strftime("%b %y") if len(monthly) else ""
    gap_note = "gaps = no valid scan" if any(v is None for v in vals) else "monthly medians"
    return f"""
    <div class='trend-block'>
      <div class='trend-title'><span>{escape(title)}</span><strong>{fmt_num(latest,decimals)}{suffix}</strong></div>
      {_sparkline_svg(vals)}
      <div class='trend-axis'><span>{first_label}</span><span>{escape(gap_note)}</span><span>{last_label}</span></div>
    </div>"""


def _value_or_dash(v: Optional[float], decimals=1) -> str:
    return "-" if v is None else f"{v:.{decimals}f}"


def _segment_card(title: str, seg: SegmentValue) -> str:
    return f"""
      <div class='seg-card'>
        <div class='seg-title'>{escape(title.upper())}</div>
        <div><span>Muscle</span> <b>{fmt_num(seg.muscle_kg,1,' kg')}</b></div>
        <div><span>Fat</span> <b>{fmt_num(seg.fat_kg,1,' kg')}</b></div>
      </div>"""


def render_report_html(data: ReportData, standalone: bool = True) -> str:
    age = age_on(data.profile, data.scan_date)
    age_text = str(age) if age is not None else "-"
    sex = data.profile.sex.title()
    name = data.profile.name.strip() or "Profile"

    bf_status, bf_cls = _status(data.body_fat_pct, data.bands.body_fat)
    mm_status, mm_cls = _status(data.muscle_pct, data.bands.muscle_pct)
    water_status, water_cls = _status(data.water_pct, data.bands.water_pct)
    bmi_status, bmi_cls = _bmi_status(data.bmi, data.bands.bmi)
    vis_status, vis_cls = _visceral_status(data.visceral_fat, data.bands.visceral_fat)
    weight_cls = bmi_cls

    bmi_lo_w = data.bands.bmi[0] * data.profile.height_m ** 2
    bmi_hi_w = data.bands.bmi[1] * data.profile.height_m ** 2

    hist_rows = []
    for _, r in data.history.iterrows():
        hist_rows.append(
            f"<tr><td>{pd.Timestamp(r['Date']).strftime('%d %b')}</td>"
            f"<td>{float(r['Weight (kg)']):.1f}</td>"
            f"<td>{float(r['Body fat %']):.1f}</td>"
            f"<td>{float(r['Muscle mass (kg)']):.1f}</td></tr>"
        )

    seg = data.segments
    la, ra = seg.get("Left Arm", SegmentValue()), seg.get("Right Arm", SegmentValue())
    ll, rl = seg.get("Left Leg", SegmentValue()), seg.get("Right Leg", SegmentValue())
    torso = seg.get("Torso", SegmentValue())
    arm_diff = _balance(la.muscle_kg, ra.muscle_kg)
    leg_diff = _balance(ll.muscle_kg, rl.muscle_kg)

    goal = data.profile.goal_weight_kg
    if goal is not None:
        diff = goal - data.weight_kg
        projected_fat = goal - data.fat_free_mass_kg
        projected_pct = projected_fat / goal * 100 if goal > 0 else None
        projection_valid = projected_fat >= 0
        goal_html = f"""
        <div class='small-label'>PERSONAL GOAL</div>
        <div class='goal-number'>{goal:.1f} kg</div>
        <div class='goal-line'>Current weight: {data.weight_kg:.2f} kg <b class='goal-delta'>{diff:+.2f} kg</b></div>
        <div class='goal-proj'>{('~'+format(projected_pct,'.1f')+'% body fat at '+format(goal,'.1f')+' kg') if projection_valid else 'Goal is below current fat-free mass'}</div>
        <div class='note'>Assuming fat-free mass is maintained. This is a mathematical projection, not a personal target.</div>
        """
        math_rows = f"""
        <div class='math-row'><span>Current weight</span><b>{data.weight_kg:.2f} kg</b></div>
        <div class='math-row'><span>Current fat mass</span><b>{data.fat_mass_kg:.2f} kg</b></div>
        <div class='math-row'><span>Current fat-free mass</span><b>{data.fat_free_mass_kg:.2f} kg</b></div>
        <div class='math-row'><span>Target weight</span><b>{goal:.1f} kg</b></div>
        <div class='math-row'><span>Projected fat mass at target</span><b>{projected_fat:.2f} kg</b></div>
        <div class='math-row'><span>Projected body fat %</span><b>{projected_pct:.1f}%</b></div>
        """ if projection_valid else f"""
        <div class='math-row'><span>Current weight</span><b>{data.weight_kg:.2f} kg</b></div>
        <div class='math-row'><span>Current fat-free mass</span><b>{data.fat_free_mass_kg:.2f} kg</b></div>
        <div class='math-row'><span>Target weight</span><b>{goal:.1f} kg</b></div>
        <div class='note warn-text'>Projection suppressed because target weight is below current fat-free mass.</div>
        """
    else:
        threshold = data.bands.bmi[1] * data.profile.height_m ** 2
        diff = threshold - data.weight_kg
        goal_html = f"""
        <div class='small-label'>PERSONAL GOAL</div>
        <div class='goal-number muted'>Not supplied</div>
        <div class='goal-line'>BMI {data.bands.bmi[1]:.1f} reference threshold:</div>
        <div class='goal-proj'>{threshold:.2f} kg <b class='goal-delta'>{diff:+.2f} kg</b></div>
        <div class='note'>Reference boundary, not a personal target.</div>
        """
        math_rows = f"""
        <div class='math-row'><span>Current weight</span><b>{data.weight_kg:.2f} kg</b></div>
        <div class='math-row'><span>Current fat mass</span><b>{data.fat_mass_kg:.2f} kg</b></div>
        <div class='math-row'><span>Current fat-free mass</span><b>{data.fat_free_mass_kg:.2f} kg</b></div>
        <div class='math-row'><span>BMI {data.bands.bmi[1]:.1f} weight</span><b>{threshold:.2f} kg</b></div>
        """

    scan_date = data.scan_date.strftime("%d %b %Y")
    scan_time = data.scan_date.strftime("%H:%M")
    seg_date = data.segment_date.strftime("%d %b %Y") if data.segment_date is not None else "No segmental scan"

    vis_value = fmt_num(data.visceral_fat, 1)
    bmr_value = f"{data.bmr:,.0f} kcal/day" if data.bmr is not None else "-"
    meta_age = f"{data.metabolic_age:.0f} y" if data.metabolic_age is not None else "-"
    vasc_age = f"{data.vascular_age:.0f} y" if data.vascular_age is not None else "-"
    icw = fmt_num(data.icw_kg, 0, " kg")
    ecw = fmt_num(data.ecw_kg, 0, " kg")
    ecw_tbw = fmt_num(data.ecw_tbw_pct, 1, "%*")

    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><style>
@page {{ size: A4 portrait; margin: 0; }}
* {{ box-sizing: border-box; }}
html,body {{ margin:0; padding:0; background:#eef3f6; font-family: Arial, Helvetica, sans-serif; color:#12466b; }}
.report-page {{ width:210mm; height:297mm; margin:0 auto; background:#fff; padding:0; overflow:hidden; position:relative; }}
.header {{ height:27mm; background:#09557f; color:#fff; padding:4.8mm 6.5mm 3mm 6.5mm; display:flex; justify-content:space-between; }}
.header h1 {{ margin:0; font-size:18pt; line-height:1; letter-spacing:.2px; }}
.header .subtitle {{ font-size:7.4pt; margin-top:2mm; opacity:.95; }}
.header-right {{ text-align:right; font-size:7.2pt; line-height:1.42; }}
.header-right .date {{ font-weight:700; font-size:8.3pt; }}
.content {{ padding:3.2mm 5.8mm 4mm; }}
.kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:2.2mm; margin-bottom:2.7mm; }}
.kpi {{ border:1px solid #b8d0df; border-radius:2.3mm; padding:2mm 2.3mm 1.6mm; height:16.8mm; position:relative; background:#fff; }}
.kpi-label {{ font-size:6.4pt; font-weight:700; color:#647b8b; }}
.kpi-value {{ margin-top:.8mm; font-size:15.2pt; font-weight:800; color:#00578c; line-height:1; }}
.kpi-sub {{ position:absolute; bottom:1.3mm; right:2mm; font-size:5.4pt; color:#73828c; }}
.dot {{ width:3.4mm; height:3.4mm; border-radius:50%; position:absolute; right:2.8mm; top:7.6mm; }}
.good {{ background:#4db66a; }} .warn {{ background:#efad2e; }} .bad {{ background:#d65c51; }} .neutral {{ background:#9aaab4; }}
.two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:2.6mm; margin-bottom:2.6mm; }}
.card {{ border:1px solid #b8d0df; border-radius:2mm; overflow:hidden; background:#fff; }}
.section-title {{ background:#086391; color:#fff; height:6.1mm; padding:1.2mm 2.5mm; font-size:7.5pt; font-weight:800; letter-spacing:.2px; }}
.comp-body {{ padding:1.3mm 2.4mm 1.2mm; height:35.7mm; }}
.comp-row {{ display:grid; grid-template-columns:1fr 31mm 24mm; align-items:center; border-bottom:1px solid #edf1f4; height:5.65mm; font-size:6.5pt; }}
.comp-row:last-child {{ border-bottom:0; }}
.comp-row b {{ text-align:right; font-size:7pt; color:#00578c; }}
.comp-row small {{ text-align:right; color:#788a95; font-size:5.5pt; }}
.bars {{ padding:1.1mm 2.4mm 1mm; height:35.7mm; }}
.bar-row {{ display:grid; grid-template-columns:22mm 1fr 21mm; gap:2mm; align-items:center; height:6.6mm; font-size:6.4pt; }}
.bar-label {{ font-weight:700; }} .bar-value {{ text-align:right; font-weight:800; font-size:7.2pt; color:#00578c; }}
.bar-note {{ font-size:5.1pt; color:#7a8a94; line-height:1.05; margin-bottom:.6mm; }}
.track {{ height:2.5mm; border-radius:2mm; background:#e6edf2; position:relative; overflow:visible; }}
.ref-zone {{ position:absolute; top:0; bottom:0; background:#c8ead2; border-radius:2mm; }}
.marker {{ position:absolute; top:50%; width:3.1mm; height:3.1mm; margin-left:-1.55mm; margin-top:-1.55mm; border-radius:50%; background:#075782; box-shadow:0 0 0 .45mm #fff; }}
.segment-card {{ margin-bottom:2.6mm; }}
.section-title.split {{ display:flex; justify-content:space-between; align-items:center; }}
.section-title.split small {{ font-size:5pt; font-weight:600; opacity:.86; letter-spacing:0; }}
.segment-wrap {{ height:57mm; padding:2mm 2.4mm 1.2mm; }}
.segment-maps {{ display:grid; grid-template-columns:1fr 1fr; gap:2.4mm; height:39.5mm; }}
.map-panel {{ border:1px solid #e0e9ee; border-radius:1.8mm; background:linear-gradient(180deg,#fbfdfe 0%,#f7fafb 100%); overflow:hidden; }}
.map-panel.muscle {{ box-shadow:inset 0 .65mm 0 #1688a6; }}
.map-panel.fat {{ box-shadow:inset 0 .65mm 0 #d79a45; }}
.map-head {{ height:5.1mm; padding:1.1mm 2mm .7mm; display:flex; justify-content:space-between; align-items:center; color:#315e78; }}
.map-head b {{ font-size:5.8pt; letter-spacing:.15px; }}
.map-head span {{ font-size:4.4pt; color:#8998a1; }}
.map-panel.muscle .map-head b {{ color:#0b718e; }} .map-panel.fat .map-head b {{ color:#a96d1e; }}
.map-content {{ display:grid; grid-template-columns:1fr 26mm 1fr; align-items:center; height:33.3mm; padding:0 1.4mm 1mm; }}
.map-side {{ height:25mm; display:flex; flex-direction:column; justify-content:space-between; }}
.map-label {{ font-size:4.8pt; line-height:1.15; color:#607988; }}
.map-label span {{ display:block; font-size:4.3pt; font-weight:800; color:#738691; margin-bottom:.55mm; }}
.map-label b {{ display:block; color:#00578c; font-size:6.1pt; }}
.map-label.right {{ text-align:right; padding-right:1.4mm; }} .map-label.left {{ text-align:left; padding-left:1.4mm; }}
.map-center {{ position:relative; height:32mm; display:flex; justify-content:center; align-items:flex-end; }}
.segment-body-svg {{ width:20mm; height:29.5mm; display:block; }}
.torso-chip {{ position:absolute; top:-.4mm; left:50%; transform:translateX(-50%); z-index:2; min-width:20mm; text-align:center; background:rgba(255,255,255,.90); border:1px solid #e2eaee; border-radius:1.3mm; padding:.55mm 1mm .5mm; line-height:1.05; }}
.torso-chip span {{ display:block; color:#718590; font-size:4pt; font-weight:800; }}
.torso-chip b {{ display:block; color:#00578c; font-size:5.8pt; margin-top:.35mm; }}
.balance-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:2.4mm; margin-top:1.6mm; }}
.balance-card {{ height:10.2mm; border:1px solid #e1e9ed; border-radius:1.5mm; padding:1.1mm 1.8mm 1mm; background:#fbfcfd; }}
.balance-head {{ display:flex; justify-content:space-between; align-items:center; font-size:4.6pt; color:#667d8b; line-height:1; }}
.balance-head b {{ color:#315e78; font-size:5pt; }}
.balance-head span {{ color:#00578c; font-weight:800; }}
.balance-values {{ display:flex; justify-content:space-between; margin-top:.9mm; font-size:4.7pt; color:#6e808b; }}
.balance-values b {{ color:#00578c; }}
.balance-track {{ display:grid; grid-template-columns:1fr .45mm 1fr; gap:.8mm; align-items:center; height:2.1mm; margin-top:.45mm; }}
.balance-track em {{ width:.45mm; height:3.2mm; background:#b7c6ce; border-radius:.4mm; }}
.balance-half {{ height:1.7mm; background:#edf2f5; border-radius:1mm; display:flex; align-items:center; overflow:hidden; }}
.balance-half.left {{ justify-content:flex-end; }} .balance-half.right {{ justify-content:flex-start; }}
.balance-half i {{ display:block; height:100%; background:#1785a5; border-radius:1mm; }}
.balance-missing {{ margin-top:2mm; font-size:4.7pt; color:#87969e; }}
.segment-note {{ text-align:center; margin-top:.75mm; font-size:4.15pt; color:#7d8d96; }}
.bottom-grid {{ display:grid; grid-template-columns:1.16fr 1.02fr .95fr; gap:2.2mm; }}
.bottom-card {{ height:119mm; }}
.history-inner {{ padding:1.3mm 2.3mm; }}
table.hist {{ width:100%; border-collapse:collapse; font-size:5.5pt; color:#315b77; margin-bottom:1mm; }}
table.hist th {{ text-align:left; font-size:5.2pt; color:#667d8d; font-weight:700; padding:.5mm 0; }}
table.hist td {{ padding:.35mm 0; }} table.hist td:not(:first-child), table.hist th:not(:first-child) {{ text-align:right; }}
.trend-block {{ margin-top:.8mm; }}
.trend-title {{ display:flex; justify-content:space-between; font-size:5.4pt; font-weight:700; color:#446b84; line-height:1; }}
.trend-title strong {{ color:#00578c; }}
.spark {{ display:block; width:100%; height:14mm; }}
.trend-axis {{ display:flex; justify-content:space-between; margin-top:-1.3mm; font-size:4.2pt; color:#8a98a1; }}
.params {{ padding:1.4mm 2.4mm; font-size:5.5pt; }}
.param-row,.math-row {{ display:flex; justify-content:space-between; line-height:1.55; }}
.param-row b,.math-row b {{ color:#00578c; }}
.subbox-title {{ background:#eef5f8; color:#0d628e; font-size:5.5pt; font-weight:800; padding:1mm 1.3mm; margin:1.3mm -1mm .8mm; border-radius:1.2mm; }}
.note-box {{ background:#eef3f6; border-radius:1.5mm; padding:1.6mm; margin-top:1.5mm; font-size:4.7pt; color:#557080; line-height:1.45; }}
.goal-inner {{ padding:1.5mm 2.5mm; font-size:5.2pt; }}
.small-label {{ font-size:5.1pt; font-weight:800; color:#607987; }}
.goal-number {{ font-size:15pt; color:#00578c; font-weight:800; line-height:1; margin:1.5mm 0 1mm; }}
.goal-number.muted {{ font-size:10pt; }}
.goal-line {{ display:flex; justify-content:space-between; align-items:center; color:#607582; }}
.goal-delta {{ color:#e67f20; margin-left:2mm; }}
.goal-proj {{ font-size:6.5pt; font-weight:800; color:#00578c; margin-top:1.1mm; }}
.note {{ font-size:4.6pt; color:#667a86; line-height:1.35; margin-top:1mm; }}
.assess {{ margin-top:1.2mm; }}
.assess-row {{ display:grid; grid-template-columns:1fr 4mm 25mm; gap:1.2mm; align-items:center; height:4.7mm; font-size:5.3pt; }}
.assess-dot {{ width:2.5mm; height:2.5mm; border-radius:50%; }}
.pill {{ border-radius:4mm; text-align:center; padding:.55mm 1mm; color:#31644a; background:#d8f0df; font-weight:700; font-size:4.8pt; }}
.pill.warn {{ color:#8e5a06; background:#ffebc3; }} .pill.bad {{ color:#873c36; background:#f5d0cd; }} .pill.neutral {{ color:#64737d; background:#e8edef; }}
.footer {{ position:absolute; left:6mm; right:6mm; bottom:3.2mm; display:flex; justify-content:space-between; font-size:4.1pt; color:#6f7f88; }}
.warn-text {{ color:#a55b16; }}
</style></head><body>
<div class='report-page'>
  <div class='header'>
    <div><h1>BODY COMPOSITION REPORT</h1><div class='subtitle'>Withings Body Scan - InBody-style one-page analysis</div></div>
    <div class='header-right'><div class='date'>{scan_date}&nbsp;&nbsp;{scan_time}</div><div>{escape(name)} | {escape(sex)} | Age {age_text} | Height {data.profile.height_m:.2f} m</div><div>Latest whole-body scan; segmental snapshot: {seg_date}</div></div>
  </div>
  <div class='content'>
    <div class='kpis'>
      <div class='kpi'><div class='kpi-label'>WEIGHT</div><div class='kpi-value'>{data.weight_kg:.2f} kg</div><div class='dot {_dot_class(weight_cls)}'></div><div class='kpi-sub'>BMI {data.bmi:.1f}</div></div>
      <div class='kpi'><div class='kpi-label'>BODY FAT</div><div class='kpi-value'>{data.body_fat_pct:.1f}%</div><div class='dot {_dot_class(bf_cls)}'></div><div class='kpi-sub'>{data.fat_mass_kg:.2f} kg</div></div>
      <div class='kpi'><div class='kpi-label'>MUSCLE MASS</div><div class='kpi-value'>{data.muscle_mass_kg:.2f} kg</div><div class='dot {_dot_class(mm_cls)}'></div><div class='kpi-sub'>{data.muscle_pct:.1f}% of weight</div></div>
      <div class='kpi'><div class='kpi-label'>VISCERAL FAT</div><div class='kpi-value'>{vis_value}</div><div class='dot {_dot_class(vis_cls)}'></div><div class='kpi-sub'>Withings index 0-20</div></div>
    </div>

    <div class='two-col'>
      <div class='card'><div class='section-title'>BODY COMPOSITION ANALYSIS</div><div class='comp-body'>
        <div class='comp-row'><span>Total Body Water</span><b>{data.water_kg:.2f} kg</b><small>{data.water_pct:.1f}%</small></div>
        <div class='comp-row'><span>Fat-Free Mass</span><b>{data.fat_free_mass_kg:.2f} kg</b><small>Weight - fat mass</small></div>
        <div class='comp-row'><span>Muscle Mass</span><b>{data.muscle_mass_kg:.2f} kg</b><small>{data.muscle_pct:.1f}%</small></div>
        <div class='comp-row'><span>Bone Mass</span><b>{data.bone_mass_kg:.2f} kg</b><small>{data.bone_pct:.1f}%</small></div>
        <div class='comp-row'><span>Body Fat Mass</span><b>{data.fat_mass_kg:.2f} kg</b><small>{data.body_fat_pct:.1f}%</small></div>
        <div class='comp-row'><span>Weight</span><b>{data.weight_kg:.2f} kg</b><small>BMI {data.bmi:.1f}</small></div>
      </div></div>
      <div class='card'><div class='section-title'>MUSCLE - FAT / OBESITY ANALYSIS</div><div class='bars'>
        {_bar(data.weight_kg,(bmi_lo_w,bmi_hi_w),(max(35,bmi_lo_w*0.65), bmi_hi_w*1.45),'Weight')}
        {_bar(data.muscle_pct,data.bands.muscle_pct,(45,95),'Muscle %')}
        {_bar(data.body_fat_pct,data.bands.body_fat,(5,45),'Body fat %')}
        {_bar(data.bmi,data.bands.bmi,(14,36),'BMI')}
        {_bar(data.water_pct,data.bands.water_pct,(35,75),'Water %')}
      </div></div>
    </div>

    <div class='card segment-card'><div class='section-title split'><span>SEGMENTAL MUSCLE & FAT ANALYSIS</span><small>Latest segmental scan: {seg_date}</small></div><div class='segment-wrap'>
      <div class='segment-maps'>
        {_segment_map(data, 'muscle')}
        {_segment_map(data, 'fat')}
      </div>
      <div class='balance-grid'>
        {_balance_card('ARM MUSCLE BALANCE', la.muscle_kg, ra.muscle_kg)}
        {_balance_card('LEG MUSCLE BALANCE', ll.muscle_kg, rl.muscle_kg)}
      </div>
      <div class='segment-note'>Colour intensity shows relative regional mass within this segmental scan only; it does not indicate a healthy/unhealthy reference range.</div>
    </div></div>

    <div class='bottom-grid'>
      <div class='card bottom-card'><div class='section-title'>BODY COMPOSITION HISTORY</div><div class='history-inner'>
        <table class='hist'><thead><tr><th>Date</th><th>Wt (kg)</th><th>Fat%</th><th>Muscle</th></tr></thead><tbody>{''.join(hist_rows)}</tbody></table>
        {_history_chart(data.monthly,'Weight (kg)','12-mo monthly median weight',' kg',1)}
        {_history_chart(data.monthly,'Body fat %','12-mo monthly median body fat','%',1)}
        {_history_chart(data.monthly,'Muscle mass (kg)','12-mo monthly median muscle',' kg',1)}
      </div></div>

      <div class='card bottom-card'><div class='section-title'>BODY SCAN PARAMETERS</div><div class='params'>
        <div class='param-row'><span>BMR</span><b>{bmr_value}</b></div>
        <div class='param-row'><span>Metabolic age</span><b>{meta_age}</b></div>
        <div class='param-row'><span>Vascular age</span><b>{vasc_age}</b></div>
        <div class='param-row'><span>ICW</span><b>{icw}</b></div>
        <div class='param-row'><span>ECW</span><b>{ecw}</b></div>
        <div class='param-row'><span>ECW/TBW</span><b>{ecw_tbw}</b></div>
        <div class='param-row'><span>Visceral fat</span><b>{vis_value} / 20</b></div>
        <div class='subbox-title'>REFERENCE BANDS</div>
        <div class='param-row'><span>Body fat</span><b>{data.bands.body_fat[0]:g}-{data.bands.body_fat[1]:g}%</b></div>
        <div class='param-row'><span>Muscle</span><b>{data.bands.muscle_pct[0]:g}-{data.bands.muscle_pct[1]:g}%</b></div>
        <div class='param-row'><span>Body water</span><b>{data.bands.water_pct[0]:g}-{data.bands.water_pct[1]:g}%</b></div>
        <div class='param-row'><span>Bone mass</span><b>{data.bands.bone_pct[0]:g}-{data.bands.bone_pct[1]:g}%</b></div>
        <div class='param-row'><span>Visceral fat</span><b>{data.bands.visceral_fat[0]:g}-{data.bands.visceral_fat[1]:g}</b></div>
        <div class='param-row'><span>BMI</span><b>{data.bands.bmi[0]:g}-{data.bands.bmi[1]:g}</b></div>
        <div class='note-box'><b>BIA consistency note</b><br>Use same time / hydration conditions.<br>Short-term shifts can be measurement noise.<br>Trend is more useful than one reading.<br>* ECW/TBW uses rounded ICW/ECW export values.</div>
      </div></div>

      <div class='card bottom-card'><div class='section-title'>GOAL & INTERPRETATION</div><div class='goal-inner'>
        {goal_html}
        <div class='subbox-title'>CURRENT ASSESSMENT</div>
        <div class='assess'>
          <div class='assess-row'><span>Body fat</span><i class='assess-dot {bf_cls}'></i><span class='pill {bf_cls if bf_cls!='good' else ''}'>{bf_status}</span></div>
          <div class='assess-row'><span>Visceral fat</span><i class='assess-dot {vis_cls}'></i><span class='pill {vis_cls if vis_cls!='good' else ''}'>{vis_status}</span></div>
          <div class='assess-row'><span>Muscle %</span><i class='assess-dot {mm_cls}'></i><span class='pill {mm_cls if mm_cls!='good' else ''}'>{mm_status}</span></div>
          <div class='assess-row'><span>BMI</span><i class='assess-dot {bmi_cls}'></i><span class='pill {bmi_cls if bmi_cls!='good' else ''}'>{bmi_status}</span></div>
          <div class='assess-row'><span>Water %</span><i class='assess-dot {water_cls}'></i><span class='pill {water_cls if water_cls!='good' else ''}'>{water_status}</span></div>
        </div>
        <div class='subbox-title'>REFERENCE MATH</div>
        {math_rows}
      </div></div>
    </div>
  </div>
  <div class='footer'><span>Withings-derived report. Muscle Mass is not strictly Skeletal Muscle Mass (SMM). No proprietary InBody Score is calculated.</span><span>Whole-body: {scan_date} &nbsp; | &nbsp; Segmental: {seg_date}</span></div>
</div></body></html>"""
    return html


def html_to_pdf_bytes(html: str) -> bytes:
    try:
        from weasyprint import HTML
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PDF export requires the optional 'weasyprint' package.") from exc
    return HTML(string=html).write_pdf()
