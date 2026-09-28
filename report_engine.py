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
    history = complete.tail(8).copy().sort_values("Date", ascending=False)
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


def _segment_total(segments: Dict[str, SegmentValue], metric: str) -> Optional[float]:
    values = [_segment_metric_value(v, metric) for v in segments.values()]
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return sum(values) if values else None


def _segment_body_svg(data: ReportData, metric: str) -> str:
    """Smooth infographic-style body silhouette with five independently colored regions.

    Colors are purely anatomical/visual and deliberately do not imply a clinical rating.
    """
    female = str(data.profile.sex).lower().startswith("f")
    if metric == "muscle":
        torso, arm, leg = "#168ac0", "#22a6c8", "#27ae9f"
        torso2, arm2, leg2 = "#0c6f9f", "#168faf", "#1c8f83"
    else:
        torso, arm, leg = "#ee8b2d", "#f3a13a", "#e66d32"
        torso2, arm2, leg2 = "#cf6f1b", "#dc8524", "#c95528"

    if female:
        torso_path = "M88 74 C96 62 105 58 120 58 C135 58 144 62 152 74 C156 92 154 108 148 125 C145 136 148 150 160 169 C150 180 136 186 120 186 C104 186 90 180 80 169 C92 150 95 136 92 125 C86 108 84 92 88 74 Z"
        pelvis_path = "M81 165 C92 176 105 182 120 182 C135 182 148 176 159 165 L154 196 C143 204 132 208 120 208 C108 208 97 204 86 196 Z"
    else:
        torso_path = "M82 75 C92 61 104 57 120 57 C136 57 148 61 158 75 C164 95 161 118 155 139 C149 158 136 174 120 183 C104 174 91 158 85 139 C79 118 76 95 82 75 Z"
        pelvis_path = "M87 176 C98 184 109 188 120 188 C131 188 142 184 153 176 L150 201 C140 207 130 211 120 211 C110 211 100 207 90 201 Z"

    uid = f"{metric}-{'f' if female else 'm'}"
    return f"""
    <svg viewBox='0 0 240 390' class='segment-body-svg' role='img' aria-label='{metric.title()} regional body schematic'>
      <defs>
        <linearGradient id='head-{uid}' x1='0' y1='0' x2='0' y2='1'>
          <stop offset='0' stop-color='#e9eff2'/><stop offset='1' stop-color='#c7d3d9'/>
        </linearGradient>
        <linearGradient id='torso-{uid}' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='{torso}'/><stop offset='1' stop-color='{torso2}'/></linearGradient>
        <linearGradient id='arm-{uid}' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='{arm}'/><stop offset='1' stop-color='{arm2}'/></linearGradient>
        <linearGradient id='leg-{uid}' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='{leg}'/><stop offset='1' stop-color='{leg2}'/></linearGradient>
        <filter id='shadow-{uid}' x='-25%' y='-20%' width='150%' height='150%'><feDropShadow dx='0' dy='2' stdDeviation='2.2' flood-color='#5d7685' flood-opacity='.18'/></filter>
      </defs>
      <g filter='url(#shadow-{uid})' stroke='#ffffff' stroke-width='2.0' stroke-linejoin='round'>
        <ellipse cx='120' cy='33' rx='21' ry='25' fill='url(#head-{uid})' stroke='#c8d4da'/>
        <path d='M109 55 C111 62 111 68 108 73 L132 73 C129 68 129 62 131 55 Z' fill='#d7e0e4' stroke='#cbd6dc'/>
        <path d='{torso_path}' fill='url(#torso-{uid})'/>
        <path d='M85 78 C68 84 57 98 51 116 L27 185 C23 197 27 205 36 205 C44 205 48 198 51 190 L75 136 C82 120 89 111 96 101 Z' fill='{arm}'/>
        <path d='M155 78 C172 84 183 98 189 116 L213 185 C217 197 213 205 204 205 C196 205 192 198 189 190 L165 136 C158 120 151 111 144 101 Z' fill='{arm}'/>
        <path d='{pelvis_path}' fill='{leg}' stroke='#ffffff'/>
        <path d='M91 198 C84 228 83 260 87 302 L83 355 C82 369 88 376 98 375 L109 370 L111 302 L117 208 Z' fill='{leg}'/>
        <path d='M149 198 C156 228 157 260 153 302 L157 355 C158 369 152 376 142 375 L131 370 L129 302 L123 208 Z' fill='{leg}'/>
        <path d='M83 353 C78 365 69 371 58 377 L100 377 L99 367 Z' fill='#dce4e8' stroke='#cbd6dc'/>
        <path d='M157 353 C162 365 171 371 182 377 L140 377 L141 367 Z' fill='#dce4e8' stroke='#cbd6dc'/>
      </g>
      <g fill='none' stroke='rgba(255,255,255,.44)' stroke-width='1.1'>
        <path d='M92 104 C109 111 131 111 148 104'/>
        <path d='M94 148 C111 154 129 154 146 148'/>
        <path d='M90 248 C98 251 104 252 110 251'/>
        <path d='M130 251 C136 252 142 251 150 248'/>
      </g>
    </svg>"""


def _segment_label(position: str, value: Optional[float], share: Optional[float], side: str = "left") -> str:
    share_text = f"{share:.1f}% of segment total" if share is not None else ""
    return f"""<div class='seg-readout {side}'>
      <span class='seg-name'>{escape(position.upper())}</span>
      <b>{fmt_num(value,1,' kg')}</b>
      <small>{share_text}</small>
    </div>"""


def _segment_analysis_panel(data: ReportData, metric: str) -> str:
    seg = data.segments
    la, ra = seg.get("Left Arm", SegmentValue()), seg.get("Right Arm", SegmentValue())
    ll, rl = seg.get("Left Leg", SegmentValue()), seg.get("Right Leg", SegmentValue())
    torso = seg.get("Torso", SegmentValue())
    getv = lambda x: _segment_metric_value(x, metric)
    total = _segment_total(seg, metric)
    share = lambda v: (float(v) / total * 100.0) if (v is not None and total and total > 0) else None
    if metric == "muscle":
        title, subtitle, cls = "SEGMENTAL MUSCLE ANALYSIS", "Withings regional muscle mass", "muscle"
        balance = f"Arm L/R difference: {fmt_num(_balance(getv(la), getv(ra)),1,'%')} &nbsp;&nbsp; | &nbsp;&nbsp; Leg L/R difference: {fmt_num(_balance(getv(ll), getv(rl)),1,'%')}"
    else:
        title, subtitle, cls = "SEGMENTAL FAT ANALYSIS", "Withings regional fat mass", "fat"
        balance = f"Segment total: {fmt_num(total,1,' kg')} &nbsp;&nbsp; • &nbsp;&nbsp; percentage = share of measured segment total"
    return f"""
      <div class='seg-panel {cls}'>
        <div class='seg-panel-head'><div><b>{title}</b><span>{subtitle}</span></div></div>
        <div class='seg-panel-body'>
          <div class='seg-side seg-left'>
            {_segment_label('Left Arm', getv(la), share(getv(la)), 'right')}
            {_segment_label('Left Leg', getv(ll), share(getv(ll)), 'right')}
          </div>
          <div class='seg-figure'>
            <div class='seg-torso'><span>TORSO</span><b>{fmt_num(getv(torso),1,' kg')}</b><small>{(f'{share(getv(torso)):.1f}% of segment total' if share(getv(torso)) is not None else '')}</small></div>
            {_segment_body_svg(data, metric)}
          </div>
          <div class='seg-side seg-right'>
            {_segment_label('Right Arm', getv(ra), share(getv(ra)), 'left')}
            {_segment_label('Right Leg', getv(rl), share(getv(rl)), 'left')}
          </div>
        </div>
        <div class='seg-panel-foot'>{balance}</div>
      </div>"""


def _history_track(values: List[float], color: str = "#1c3f52", width: int = 620, height: int = 86) -> str:
    if not values:
        return f"<svg viewBox='0 0 {width} {height}' class='history-svg'></svg>"
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return f"<svg viewBox='0 0 {width} {height}' class='history-svg'></svg>"
    lo, hi = float(finite.min()), float(finite.max())
    if math.isclose(lo, hi):
        lo -= 1.0; hi += 1.0
    pad = max((hi-lo)*0.34, 0.4)
    lo -= pad; hi += pad
    n = len(arr)
    xs = np.linspace(24, width-24, n) if n > 1 else np.array([width/2])
    def yy(v):
        return height - 18 - (v-lo)/(hi-lo)*(height-39)
    pts=[]; dots=[]; labels=[]
    for x,v in zip(xs,arr):
        y=yy(float(v)); pts.append(f"{x:.1f},{y:.1f}")
        dots.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3.5' fill='{color}'/>")
        labels.append(f"<text x='{x:.1f}' y='{max(11,y-8):.1f}' text-anchor='middle' class='hval'>{v:.1f}</text>")
    grid = ''.join(f"<line x1='{x:.1f}' x2='{x:.1f}' y1='18' y2='{height-13}' class='hgrid'/>" for x in xs)
    return f"""<svg viewBox='0 0 {width} {height}' class='history-svg' preserveAspectRatio='none'>
      {grid}<line x1='12' x2='{width-12}' y1='{height-14}' y2='{height-14}' class='hbase'/>
      <polyline points='{' '.join(pts)}' fill='none' stroke='{color}' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'/>
      {''.join(dots)}{''.join(labels)}
    </svg>"""


def _history_panel(data: ReportData) -> str:
    h = data.history.copy().sort_values("Date")
    if h.empty:
        return "<div class='history-empty'>No complete scan history.</div>"
    # Keep the last eight actual complete scans, matching the compact InBody-style history layout.
    h = h.tail(8)
    dates=[pd.Timestamp(d).strftime('%d %b') for d in h['Date']]
    weights=[float(v) for v in h['Weight (kg)']]
    muscle=[float(v) for v in h['Muscle mass (kg)']]
    fat=[float(v) for v in h['Body fat %']]

    def latest_monthly(col, suffix):
        vals=[float(v) for v in data.monthly[col].tolist() if not pd.isna(v)]
        return (f"{vals[-1]:.1f}{suffix}" if vals else "-")

    def row(label, sublabel, vals, median_text):
        return f"""<div class='ih-row'>
          <div class='ih-label'><b>{label}</b><span>{sublabel}</span><small>Latest monthly median&nbsp; {median_text}</small></div>
          <div class='ih-plot'>{_history_track(vals)}</div>
        </div>"""

    date_cells=''.join(f"<span><b>{escape(d.split()[0])}</b><small>{escape(d.split()[1]) if len(d.split())>1 else ''}</small></span>" for d in dates)
    return f"""
      <div class='ih-wrap'>
        {row('Weight','kg',weights,latest_monthly('Weight (kg)',' kg'))}
        {row('Muscle Mass','kg',muscle,latest_monthly('Muscle mass (kg)',' kg'))}
        {row('Body Fat','%',fat,latest_monthly('Body fat %','%'))}
        <div class='ih-dates'><div></div><div class='ih-date-grid' style='grid-template-columns:repeat({len(dates)},1fr)'>{date_cells}</div></div>
        <div class='ih-note'>Recent {len(dates)} complete whole-body scans shown. Monthly medians remain gap-aware and are not interpolated.</div>
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
.segment-wrap {{ height:66mm; padding:1.8mm 2.2mm 1.4mm; }}
.segment-panels {{ display:grid; grid-template-columns:1fr 1fr; gap:2.4mm; height:61.5mm; }}
.seg-panel {{ border:1px solid #d7e2e8; background:#f3f7f9; min-width:0; position:relative; overflow:hidden; }}
.seg-panel.muscle {{ box-shadow:inset 0 .8mm 0 #168ac0; }}
.seg-panel.fat {{ box-shadow:inset 0 .8mm 0 #ee8b2d; }}
.seg-panel-head {{ height:8.2mm; padding:1.7mm 2mm 1.1mm; background:#e8f0f4; display:flex; align-items:center; }}
.seg-panel-head b {{ display:block; font-size:6.9pt; color:#395a6c; line-height:1; }}
.seg-panel.muscle .seg-panel-head b {{ color:#08739e; }} .seg-panel.fat .seg-panel-head b {{ color:#b96619; }}
.seg-panel-head span {{ display:block; margin-top:.7mm; font-size:4.4pt; color:#788c98; }}
.seg-panel-body {{ height:47mm; display:grid; grid-template-columns:1fr 30mm 1fr; align-items:center; padding:.7mm 1.5mm 0; }}
.seg-side {{ height:35.5mm; display:flex; flex-direction:column; justify-content:space-between; padding-top:4mm; padding-bottom:3mm; }}
.seg-readout {{ color:#5f7786; line-height:1.1; }}
.seg-readout.right {{ text-align:right; padding-right:1.2mm; }} .seg-readout.left {{ text-align:left; padding-left:1.2mm; }}
.seg-name {{ display:block; font-size:4.4pt; font-weight:800; color:#607887; letter-spacing:.1px; }}
.seg-readout b {{ display:block; font-size:7.4pt; margin-top:.7mm; color:#154f70; }}
.seg-panel.muscle .seg-readout b {{ color:#08739e; }} .seg-panel.fat .seg-readout b {{ color:#b96619; }}
.seg-readout small {{ display:block; margin-top:.5mm; font-size:3.8pt; color:#8a99a2; white-space:nowrap; }}
.seg-figure {{ position:relative; height:46mm; display:flex; justify-content:center; align-items:flex-end; }}
.segment-body-svg {{ width:29mm; height:45mm; display:block; }}
.seg-torso {{ position:absolute; top:1mm; left:50%; transform:translateX(-50%); z-index:3; width:29mm; text-align:center; line-height:1.03; background:rgba(255,255,255,.88); border:1px solid rgba(197,211,219,.85); padding:.7mm .8mm .6mm; }}
.seg-torso span {{ display:block; font-size:4pt; color:#6f838e; font-weight:800; }}
.seg-torso b {{ display:block; font-size:7.2pt; margin-top:.45mm; color:#154f70; }}
.seg-panel.muscle .seg-torso b {{ color:#08739e; }} .seg-panel.fat .seg-torso b {{ color:#b96619; }}
.seg-torso small {{ display:block; font-size:3.7pt; color:#8a99a2; margin-top:.4mm; }}
.seg-panel-foot {{ height:5.8mm; border-top:1px solid #dce6eb; padding:1.25mm 1.5mm 0; text-align:center; font-size:4.2pt; color:#6f838e; white-space:nowrap; }}
.bottom-grid {{ display:grid; grid-template-columns:1.42fr .91fr .91fr; gap:2.2mm; }}
.bottom-card {{ height:110mm; }}
.history-inner {{ padding:1.3mm 1.5mm 1mm; }}
.ih-wrap {{ width:100%; }}
.ih-row {{ display:grid; grid-template-columns:24mm 1fr; min-height:23.4mm; border-bottom:1px solid #d9e1e6; background:#fff; }}
.ih-label {{ background:#e2eaee; padding:2.2mm 1.5mm 1.3mm; color:#394f5a; }}
.ih-label b {{ display:block; font-size:7pt; line-height:1; }}
.ih-label span {{ display:block; font-size:4.2pt; margin-top:.5mm; color:#6f818b; }}
.ih-label small {{ display:block; font-size:3.8pt; margin-top:1.3mm; line-height:1.22; color:#6f818b; }}
.ih-plot {{ padding:.6mm .5mm .2mm 1mm; overflow:hidden; }}
.history-svg {{ display:block; width:100%; height:21.5mm; }}
.history-svg .hgrid {{ stroke:#eef2f4; stroke-width:1; }} .history-svg .hbase {{ stroke:#d8e0e4; stroke-width:1; }}
.history-svg .hval {{ fill:#2b3e48; font-size:17px; font-weight:700; font-family:Arial,Helvetica,sans-serif; }}
.ih-dates {{ display:grid; grid-template-columns:24mm 1fr; min-height:8.8mm; }}
.ih-date-grid {{ display:grid; align-items:start; text-align:center; color:#697d88; padding:.8mm .3mm 0 1mm; }}
.ih-date-grid span {{ display:block; justify-self:center; white-space:nowrap; line-height:1.05; font-size:3.7pt; }}
.ih-date-grid b {{ display:block; font-size:4.2pt; color:#546a76; }}
.ih-date-grid small {{ display:block; margin-top:.25mm; font-size:3.6pt; color:#7f9099; }}
.ih-note {{ margin-top:1.1mm; padding:1.2mm 1.4mm; background:#eef4f7; font-size:4.2pt; color:#687d89; line-height:1.32; }}
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

    <div class='card segment-card'><div class='section-title split'><span>SEGMENTAL ANALYSIS</span><small>Latest segmental scan: {seg_date}</small></div><div class='segment-wrap'>
      <div class='segment-panels'>
        {_segment_analysis_panel(data, 'muscle')}
        {_segment_analysis_panel(data, 'fat')}
      </div>
    </div></div>

    <div class='bottom-grid'>
      <div class='card bottom-card'><div class='section-title'>BODY COMPOSITION HISTORY</div><div class='history-inner'>
        {_history_panel(data)}
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
