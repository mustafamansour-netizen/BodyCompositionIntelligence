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


ENGINE_BUILD = "V8.1"

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
    history_daily_rule: str = "Earliest complete scan"
    history_points: int = 8
    profile_id: Optional[str] = None
    journey_start_date: Optional[date] = None
    starting_weight_kg: Optional[float] = None
    target_body_fat_pct: Optional[float] = None
    display_units: str = "Metric"  # Metric / US Customary


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
    pwv_mps: Optional[float]
    scan_heart_rate_bpm: Optional[float]
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

CANONICAL_WEIGHT_COLUMNS = [
    "Weight (kg)", "Fat mass (kg)", "Bone mass (kg)", "Muscle mass (kg)", "Hydration (kg)"
]
OTHER_REQUIRED = {"type", "date", "value", "unit", "position"}

_MASS_TO_KG = {
    "kg": 1.0, "kgs": 1.0, "kilogram": 1.0, "kilograms": 1.0,
    "lb": 0.45359237, "lbs": 0.45359237, "pound": 0.45359237, "pounds": 0.45359237,
}


def _clean_unit(value: object) -> str:
    u = str(value or "").strip().lower().replace(".", "")
    return re.sub(r"\s+", " ", u)


def _mass_factor_to_kg(unit: object) -> Optional[float]:
    return _MASS_TO_KG.get(_clean_unit(unit))


def _column_unit(col: object) -> Optional[str]:
    s = str(col)
    m = re.search(r"\(([^)]+)\)\s*$", s)
    return _clean_unit(m.group(1)) if m else None


def _match_weight_column(columns: Iterable[object], concept: str) -> Tuple[Optional[str], Optional[str]]:
    patterns = {
        "weight": ["weight"],
        "fat": ["fat mass", "fatmass"],
        "bone": ["bone mass", "bonemass"],
        "muscle": ["muscle mass", "musclemass"],
        "water": ["hydration", "body water", "water mass"],
    }
    candidates = []
    for col in columns:
        n = _norm(col)
        if any(p in n for p in patterns[concept]):
            unit = _column_unit(col)
            if unit in _MASS_TO_KG:
                candidates.append((str(col), unit))
    if not candidates:
        return None, None
    # Prefer exact/common labels and kg/lb explicit headers.
    return candidates[0]


def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    last_err = None
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "cp1252", "latin1"):
        try:
            text = raw.decode(encoding)
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
    return sorted(candidates, key=lambda x: (x.count("/"), len(x)))[0]


def _optional_csv(zf: zipfile.ZipFile, basename: str) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    name = _find_member(zf, basename)
    if name is None:
        return None, None
    try:
        return name, _read_csv_bytes(zf.read(name))
    except Exception:
        return name, None


def _append_optional_measurements(other_df: pd.DataFrame, pwv_df: Optional[pd.DataFrame], bp_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    frames = [other_df.copy()]
    if pwv_df is not None and {"date", "value"}.issubset(pwv_df.columns):
        x = pwv_df[["date", "value"]].copy()
        x["type"] = "Pulse Wave Velocity"
        x["unit"] = "m/s"
        x["position"] = ""
        frames.append(x[["type", "date", "value", "unit", "position"]])
    if bp_df is not None and {"Date", "Heart rate"}.issubset(bp_df.columns):
        x = bp_df[["Date", "Heart rate"]].copy().rename(columns={"Date":"date", "Heart rate":"value"})
        x["type"] = "Scan heart rate"
        x["unit"] = "bpm"
        x["position"] = ""
        frames.append(x[["type", "date", "value", "unit", "position"]])
    return pd.concat(frames, ignore_index=True, sort=False)


def load_withings_zip(raw_zip: bytes, source_mass_unit_override: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
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
        pwv_name, pwv_df = _optional_csv(zf, "pwv.csv")
        bp_name, bp_df = _optional_csv(zf, "bp.csv")
        other_df = _append_optional_measurements(other_df, pwv_df, bp_df)
        meta = {
            "weight_member": weight_name, "other_member": other_name,
            "pwv_member": pwv_name, "bp_member": bp_name,
            "pwv_records": 0 if pwv_df is None else int(len(pwv_df)),
            "bp_records": 0 if bp_df is None else int(len(bp_df)),
        }
    return normalize_frames(weight_df, other_df, meta, source_mass_unit_override=source_mass_unit_override)


def load_withings_csvs(
    weight_raw: bytes,
    other_raw: bytes,
    pwv_raw: Optional[bytes] = None,
    bp_raw: Optional[bytes] = None,
    source_mass_unit_override: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    weight_df = _read_csv_bytes(weight_raw)
    other_df = _read_csv_bytes(other_raw)
    pwv_df = _read_csv_bytes(pwv_raw) if pwv_raw else None
    bp_df = _read_csv_bytes(bp_raw) if bp_raw else None
    other_df = _append_optional_measurements(other_df, pwv_df, bp_df)
    return normalize_frames(
        weight_df, other_df,
        {"weight_member":"weight.csv", "other_member":"other.csv", "pwv_member":"pwv.csv" if pwv_raw else None, "bp_member":"bp.csv" if bp_raw else None},
        source_mass_unit_override=source_mass_unit_override,
    )


def normalize_frames(weight_df: pd.DataFrame, other_df: pd.DataFrame, meta=None, source_mass_unit_override: Optional[str] = None):
    """Normalize a Withings export into canonical metric columns.

    Import/source units and display units are deliberately independent. All body-mass
    analytics use kg internally. Explicit kg/lb headers and other.csv row units are
    converted here once; rendering may later show either Metric or US Customary units.
    """
    meta = dict(meta or {})
    if not OTHER_REQUIRED.issubset(set(other_df.columns)):
        missing_o = sorted(OTHER_REQUIRED - set(other_df.columns))
        raise ValueError("Missing other.csv columns: " + ", ".join(missing_o))

    weight_df = weight_df.copy()
    other_df = other_df.copy()
    override = _clean_unit(source_mass_unit_override) if source_mass_unit_override else None
    if override in {"auto", ""}:
        override = None
    if override and override not in _MASS_TO_KG:
        raise ValueError("Source unit override must be kg or lb.")

    source_map = {}
    concept_to_canon = {
        "weight":"Weight (kg)", "fat":"Fat mass (kg)", "bone":"Bone mass (kg)",
        "muscle":"Muscle mass (kg)", "water":"Hydration (kg)",
    }
    canonical = pd.DataFrame()
    # Preserve date/comments then rewrite the five mass columns canonically.
    date_col = next((c for c in weight_df.columns if _norm(c) == "date"), None)
    if date_col is None:
        raise ValueError("Could not identify the Date column in weight.csv.")
    canonical["Date"] = weight_df[date_col]
    if "Comments" in weight_df.columns:
        canonical["Comments"] = weight_df["Comments"]

    ambiguous = []
    for concept, canon in concept_to_canon.items():
        col, unit = _match_weight_column(weight_df.columns, concept)
        if col is None:
            # Fallback for a unit-less concept column only when an override was explicitly supplied.
            possible = [c for c in weight_df.columns if any(p in _norm(c) for p in ({
                "weight":["weight"], "fat":["fat mass"], "bone":["bone mass"],
                "muscle":["muscle mass"], "water":["hydration", "body water", "water mass"]
            }[concept]))]
            if possible and override:
                col, unit = str(possible[0]), override
            else:
                ambiguous.append(canon)
                continue
        factor = _mass_factor_to_kg(unit or override)
        if factor is None:
            ambiguous.append(canon)
            continue
        canonical[canon] = pd.to_numeric(weight_df[col], errors="coerce") * factor
        source_map[canon] = {"source_column": col, "source_unit": unit or override, "factor_to_kg": factor}

    if ambiguous:
        raise ValueError(
            "Could not determine source mass units for: " + ", ".join(ambiguous) + ". "
            "Use the Source unit override in Advanced import settings if this export omits units."
        )
    weight_df = canonical

    weight_df["Date"] = pd.to_datetime(weight_df["Date"], errors="coerce")
    try:
        if getattr(weight_df["Date"].dt, "tz", None) is not None:
            weight_df["Date"] = weight_df["Date"].dt.tz_localize(None)
    except Exception:
        pass

    other_df["date"] = pd.to_datetime(other_df["date"], errors="coerce")
    try:
        if getattr(other_df["date"].dt, "tz", None) is not None:
            other_df["date"] = other_df["date"].dt.tz_localize(None)
    except Exception:
        pass
    other_df["value_numeric"] = pd.to_numeric(other_df["value"], errors="coerce")

    # Infer a source-mass fallback from weight.csv when all detected mass headers agree.
    detected_units = {str(v.get("source_unit")) for v in source_map.values() if v.get("source_unit")}
    inferred_mass_unit = next(iter(detected_units)) if len(detected_units) == 1 else None

    # Normalize mass-valued other.csv rows (segmental mass, ICW/ECW) to kg.
    def mass_like(t: object) -> bool:
        n = _norm(t)
        return ("segment" in n and "mass" in n) or n in {"icw", "ecw"} or "intracellular water" in n or "extracellular water" in n
    converted = 0
    for idx, row in other_df.iterrows():
        if not mass_like(row.get("type")) or pd.isna(row.get("value_numeric")):
            continue
        u = _clean_unit(row.get("unit"))
        factor = _mass_factor_to_kg(u)
        if factor is None and (override or inferred_mass_unit):
            factor = _mass_factor_to_kg(override or inferred_mass_unit)
        if factor is not None:
            other_df.at[idx, "value_numeric"] = float(row["value_numeric"]) * factor
            other_df.at[idx, "unit"] = "kg"
            converted += 1

    weight_df = weight_df.dropna(subset=["Date", "Weight (kg)"]).sort_values("Date").reset_index(drop=True)
    other_df = other_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    meta.update({
        "weight_records": len(weight_df),
        "other_records": len(other_df),
        "other_types": sorted(other_df["type"].dropna().astype(str).unique().tolist()),
        "positions": sorted(other_df["position"].dropna().astype(str).unique().tolist()),
        "source_mass_columns": source_map,
        "internal_mass_unit": "kg",
        "inferred_source_mass_unit": inferred_mass_unit,
        "normalized_other_mass_rows": converted,
    })
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
    if "pulse wave velocity" in s or s == "pwv":
        return "pwv"
    if "scan heart rate" in s:
        return "scan_heart_rate"
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


def _nearest_metric(other_df: pd.DataFrame, kind: str, target: pd.Timestamp, tolerance_minutes: int = 5) -> Optional[float]:
    if other_df.empty:
        return None
    mask = other_df["type"].map(_metric_kind).eq(kind) & other_df["value_numeric"].notna()
    subset = other_df.loc[mask].copy()
    if subset.empty:
        return None
    subset["delta"] = (subset["date"] - target).abs()
    subset = subset[subset["delta"] <= pd.Timedelta(minutes=tolerance_minutes)]
    if subset.empty:
        return None
    row = subset.sort_values("delta").iloc[0]
    return float(row["value_numeric"])


def _find_latest_segment_snapshot(other_df: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Dict[str, SegmentValue], Dict[str, object]]:
    """Return the latest usable segmental snapshot plus prior complete snapshot metadata.

    The latest snapshot follows the existing rule: prefer the latest snapshot with all
    five muscle and fat regions, otherwise use the latest snapshot with the best
    coverage.  For change arrows we only compare against the previous *fully complete*
    segmental snapshot so that a missing limb cannot look like a gain/loss.
    """
    seg = other_df.copy()
    seg["kind"] = seg["type"].map(_metric_kind)
    seg["position_canon"] = seg["position"].map(_canonical_position)
    seg = seg[
        seg["kind"].isin(["segment_muscle", "segment_fat", "segment_ffm"])
        & seg["position_canon"].notna()
        & seg["value_numeric"].notna()
    ].copy()

    positions = ["Left Arm", "Right Arm", "Torso", "Left Leg", "Right Leg"]
    target_positions = set(positions)
    empty = {p: SegmentValue() for p in positions}
    if seg.empty:
        return None, empty, {"segment_candidates": 0}

    seg["scan_key"] = seg["date"].dt.floor("min")
    candidates = []
    for scan_key, g in seg.groupby("scan_key"):
        mus_pos = set(g.loc[g["kind"] == "segment_muscle", "position_canon"])
        fat_pos = set(g.loc[g["kind"] == "segment_fat", "position_canon"])
        complete = len(target_positions & mus_pos & fat_pos)
        candidates.append((pd.Timestamp(scan_key), complete, len(g)))

    def build_snapshot(scan_key: pd.Timestamp) -> Dict[str, SegmentValue]:
        g = seg[seg["scan_key"] == scan_key]
        out: Dict[str, SegmentValue] = {}
        for pos in positions:
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
        return out

    full = [x for x in candidates if x[1] == 5]
    if full:
        scan_key = max(full, key=lambda x: x[0])[0]
    else:
        best_coverage = max(x[1] for x in candidates)
        scan_key = max([x for x in candidates if x[1] == best_coverage], key=lambda x: x[0])[0]

    out = build_snapshot(scan_key)
    prior_full = [x for x in full if x[0] < scan_key]
    previous_segment_date = None
    previous_segments = None
    if prior_full:
        previous_segment_date = max(prior_full, key=lambda x: x[0])[0]
        previous_segments = build_snapshot(previous_segment_date)

    return pd.Timestamp(scan_key), out, {
        "segment_candidates": len(candidates),
        "segment_complete_positions": next((x[1] for x in candidates if x[0] == scan_key), 0),
        "previous_segment_date": previous_segment_date,
        "previous_segments": previous_segments,
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


def _daily_history(complete: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Return one representative complete scan per calendar day.

    Supported rules:
    - Earliest complete scan
    - Latest complete scan
    - Daily median

    Daily median is calculated only from complete scans and only for the numeric body-
    composition fields used by the report. The 12-month monthly medians remain based on
    all complete scans and are unaffected by this display rule.
    """
    df = complete.copy().sort_values("Date")
    df["_scan_day"] = df["Date"].dt.floor("D")
    rule_norm = str(rule or "").strip().lower()

    if rule_norm.startswith("latest"):
        daily = df.groupby("_scan_day", group_keys=False).tail(1).copy()
    elif "median" in rule_norm:
        cols = ["Weight (kg)", "Fat mass (kg)", "Bone mass (kg)", "Muscle mass (kg)", "Hydration (kg)"]
        daily = df.groupby("_scan_day", as_index=False)[cols].median()
        daily = daily.rename(columns={"_scan_day": "Date"})
        return daily.sort_values("Date").reset_index(drop=True)
    else:
        daily = df.groupby("_scan_day", group_keys=False).head(1).copy()

    return daily.drop(columns=["_scan_day"]).sort_values("Date").reset_index(drop=True)


def _period_trend(daily: pd.DataFrame, scan_date: pd.Timestamp, days: int) -> Dict[str, object]:
    """Compare the current scan with a representative daily scan near a period cutoff.

    The label remains 30-day/90-day for usability, while ``span_days`` records the
    actual available interval. A comparison is suppressed when the available history
    is too short to represent the requested period.
    """
    if daily.empty:
        return {}
    d = daily.copy().sort_values("Date")
    d = d[d["Date"].dt.floor("D") < scan_date.floor("D")].copy()
    if d.empty:
        return {}
    target = scan_date.floor("D") - pd.Timedelta(days=days)
    d["_distance"] = (d["Date"].dt.floor("D") - target).abs().dt.days
    candidate = d.sort_values(["_distance", "Date"]).iloc[0]
    span = int((scan_date.floor("D") - pd.Timestamp(candidate["Date"]).floor("D")).days)
    min_span = 21 if days == 30 else 60 if days == 90 else max(1, int(days * .65))
    max_span = 45 if days == 30 else 120 if days == 90 else int(days * 1.35)
    if span < min_span or span > max_span:
        return {}
    w = float(candidate["Weight (kg)"])
    fm = float(candidate["Fat mass (kg)"])
    mm = float(candidate["Muscle mass (kg)"])
    bf = fm / w * 100 if w else math.nan
    return {
        "baseline_date": pd.Timestamp(candidate["Date"]),
        "span_days": span,
        "weight_kg": w,
        "fat_mass_kg": fm,
        "muscle_mass_kg": mm,
        "body_fat_pct": bf,
    }


def _time_consistency(complete: pd.DataFrame, n: int = 10) -> Dict[str, object]:
    """Describe recent time-of-day consistency without making a clinical judgment."""
    if complete.empty:
        return {"label":"No data", "class":"neutral", "scan_count":0}
    recent = complete.sort_values("Date").tail(max(2, n)).copy()
    mins = recent["Date"].dt.hour * 60 + recent["Date"].dt.minute + recent["Date"].dt.second / 60.0
    if len(mins) < 2:
        return {"label":"Single scan", "class":"neutral", "scan_count":int(len(mins)), "median_deviation_hours":0.0, "max_deviation_hours":0.0}
    angles = np.asarray(mins, dtype=float) / 1440.0 * 2 * np.pi
    mean_angle = math.atan2(float(np.sin(angles).mean()), float(np.cos(angles).mean()))
    if mean_angle < 0:
        mean_angle += 2 * np.pi
    center = mean_angle / (2*np.pi) * 1440.0
    diffs = np.abs(((np.asarray(mins, dtype=float) - center + 720.0) % 1440.0) - 720.0)
    med_h = float(np.median(diffs) / 60.0)
    max_h = float(np.max(diffs) / 60.0)
    if med_h <= 1.0 and max_h <= 2.0:
        label, cls = "Consistent", "good"
    elif med_h <= 2.0 and max_h <= 4.0:
        label, cls = "Moderate variation", "warn"
    else:
        label, cls = "Variable timing", "warn"
    return {"label":label, "class":cls, "scan_count":int(len(mins)), "median_deviation_hours":med_h, "max_deviation_hours":max_h}




def _scan_quality(complete: pd.DataFrame, other_df: pd.DataFrame, scan_date: pd.Timestamp, consistency: Dict[str, object]) -> Dict[str, object]:
    """Conservative data-quality context for short-term comparisons.

    This does not claim medical accuracy. It combines completeness, timing consistency,
    plausible composition ranges and unusually large short-term water shifts. BIA Error
    codes are counted for diagnostics but are not interpreted because Withings does not
    publish a stable public mapping for every export code.
    """
    issues = []
    cls = "good"
    latest = complete.iloc[-1]
    w = float(latest["Weight (kg)"])
    fm = float(latest["Fat mass (kg)"])
    mm = float(latest["Muscle mass (kg)"])
    water = float(latest["Hydration (kg)"])
    if not (25 <= w <= 350):
        issues.append("weight outside plausibility range")
    if fm < 0 or mm < 0 or water < 0 or fm > w or mm > w or water > w:
        issues.append("composition values outside plausibility range")
    if len(complete) >= 2:
        prev = complete.iloc[-2]
        if pd.Timestamp(prev["Date"]).floor("D") >= scan_date.floor("D") - pd.Timedelta(days=3):
            water_shift = abs(float(latest["Hydration (kg)"]) - float(prev["Hydration (kg)"]))
            if water_shift >= 2.5:
                issues.append(f"large short-term body-water shift ({water_shift:.1f} kg)")
    timing_label = str((consistency or {}).get("label", ""))
    if timing_label in {"Variable timing", "Moderate variation"}:
        issues.append("recent measurement times vary")
    bia = other_df[other_df["type"].map(_norm).eq("bia error") & other_df["value_numeric"].notna()].copy()
    bia_near = bia[(bia["date"] - scan_date).abs() <= pd.Timedelta(minutes=5)] if not bia.empty else bia
    bia_codes = [float(v) for v in bia_near["value_numeric"].tolist()]
    if issues:
        cls = "warn"
        label = "Variable"
    else:
        label = "Good"
    if len(complete) < 2:
        label, cls = "Limited data", "neutral"
    return {
        "label": label,
        "class": cls,
        "issues": issues,
        "bia_error_codes_near_scan": bia_codes,
        "note": "Short-term BIA changes should be interpreted with consistent measurement conditions.",
    }

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
    daily = _daily_history(complete, profile.history_daily_rule)
    points = int(max(3, min(16, profile.history_points or 8)))
    history = daily.tail(points).copy().sort_values("Date", ascending=False)
    history["Body fat %"] = history["Fat mass (kg)"] / history["Weight (kg)"] * 100
    monthly = _monthly_history(complete, scan_date)

    previous_days = daily[daily["Date"].dt.floor("D") < scan_date.floor("D")].sort_values("Date")
    previous = previous_days.iloc[-1] if not previous_days.empty else None

    last30 = complete[(complete["Date"] >= scan_date - pd.Timedelta(days=30)) & (complete["Date"] <= scan_date)].copy()
    last30_bf = last30["Fat mass (kg)"] / last30["Weight (kg)"] * 100 if not last30.empty else pd.Series(dtype=float)

    trend30 = _period_trend(daily, scan_date, 30)
    trend90 = _period_trend(daily, scan_date, 90)
    consistency = _time_consistency(complete, 10)
    quality = _scan_quality(complete, other_df, scan_date, consistency)

    diag = dict(diagnostics or {})
    if previous is not None:
        prev_weight = float(previous["Weight (kg)"])
        prev_fat_mass = float(previous["Fat mass (kg)"])
        prev_muscle = float(previous["Muscle mass (kg)"])
        prev_bf = prev_fat_mass / prev_weight * 100
        diag.update({
            "previous_scan_date": pd.Timestamp(previous["Date"]),
            "previous_weight_kg": prev_weight,
            "previous_fat_mass_kg": prev_fat_mass,
            "previous_muscle_mass_kg": prev_muscle,
            "previous_body_fat_pct": prev_bf,
        })
    if not last30.empty:
        diag.update({
            "median30_weight_kg": float(last30["Weight (kg)"].median()),
            "median30_fat_mass_kg": float(last30["Fat mass (kg)"].median()),
            "median30_muscle_mass_kg": float(last30["Muscle mass (kg)"].median()),
            "median30_body_fat_pct": float(last30_bf.median()),
            "median30_scan_count": int(len(last30)),
        })

    for label, trend in (("trend30", trend30), ("trend90", trend90)):
        if trend:
            diag[label] = trend
            diag[f"{label}_weight_delta"] = float(latest["Weight (kg)"]) - float(trend["weight_kg"])
            diag[f"{label}_fat_mass_delta"] = float(latest["Fat mass (kg)"]) - float(trend["fat_mass_kg"])
            diag[f"{label}_muscle_delta"] = float(latest["Muscle mass (kg)"]) - float(trend["muscle_mass_kg"])
            current_bf = float(latest["Fat mass (kg)"]) / float(latest["Weight (kg)"]) * 100
            diag[f"{label}_body_fat_delta"] = current_bf - float(trend["body_fat_pct"])

    rate_source = trend30 if trend30 else trend90
    rate_key = "trend30_weight_delta" if trend30 else "trend90_weight_delta"
    if rate_source and diag.get(rate_key) is not None and rate_source.get("span_days"):
        diag["weight_rate_per_week"] = float(diag[rate_key]) / float(rate_source["span_days"]) * 7.0
        diag["weight_rate_span_days"] = int(rate_source["span_days"])

    diag["measurement_consistency"] = consistency
    diag["scan_quality"] = quality

    vf = other_df.copy()
    vf = vf[vf["type"].map(_metric_kind).eq("visceral_fat") & vf["value_numeric"].notna()]
    vf = vf[vf["date"] <= scan_date + pd.Timedelta(hours=18)].sort_values("date")
    if len(vf) >= 2:
        diag["previous_visceral_fat"] = float(vf.iloc[-2]["value_numeric"])
        diag["previous_visceral_date"] = pd.Timestamp(vf.iloc[-2]["date"])

    diag.update(seg_diag)
    diag["complete_scans"] = int(len(complete))
    diag["latest_complete_scan"] = scan_date
    diag["history_daily_rule"] = profile.history_daily_rule
    diag["history_points"] = points

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
        pwv_mps=_latest_metric(other_df, "pwv", scan_date),
        scan_heart_rate_bpm=_nearest_metric(other_df, "scan_heart_rate", scan_date, tolerance_minutes=5),
        segments=segments,
        history=history,
        monthly=monthly,
        diagnostics=diag,
    )


# -----------------------------
# Render helpers
# -----------------------------



LB_PER_KG = 2.20462262185


def _is_us(profile: Profile) -> bool:
    return str(getattr(profile, "display_units", "Metric")).lower().startswith("us")


def display_mass_value(kg: Optional[float], profile: Profile) -> Optional[float]:
    if kg is None or (isinstance(kg, float) and math.isnan(kg)):
        return None
    return float(kg) * LB_PER_KG if _is_us(profile) else float(kg)


def mass_unit(profile: Profile) -> str:
    return "lb" if _is_us(profile) else "kg"


def fmt_mass(kg: Optional[float], profile: Profile, decimals: int = 1) -> str:
    v = display_mass_value(kg, profile)
    if v is None:
        return "-"
    return f"{v:.{decimals}f} {mass_unit(profile)}"


def fmt_mass_delta(kg_delta: Optional[float], profile: Profile, decimals: int = 1) -> str:
    if kg_delta is None or (isinstance(kg_delta, float) and math.isnan(kg_delta)):
        return "-"
    v = float(kg_delta) * (LB_PER_KG if _is_us(profile) else 1.0)
    arrow = "↑" if v > 0 else ("↓" if v < 0 else "→")
    return f"{arrow} {abs(v):.{decimals}f} {mass_unit(profile)}"


def fmt_height(height_m: float, profile: Profile) -> str:
    if not _is_us(profile):
        return f"{height_m*100:.0f} cm"
    total_inches = height_m / 0.0254
    feet = int(total_inches // 12)
    inches = total_inches - feet * 12
    # Whole inches are easier to scan in a report; preserve one decimal only when materially non-integer.
    if abs(inches - round(inches)) < 0.15:
        return f"{feet} ft {round(inches):.0f} in"
    return f"{feet} ft {inches:.1f} in"

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


def _bar(value: float, ref: Tuple[float, float], domain: Tuple[float, float], label: str, unit_suffix: str = "") -> str:
    dlo, dhi = domain
    rlo, rhi = ref
    def pct(x):
        return max(0.0, min(100.0, (x - dlo) / (dhi - dlo) * 100.0))
    rp1, rp2, vp = pct(rlo), pct(rhi), pct(value)
    pct_suffix = "%" if "%" in label else ""
    return f"""
    <div class='bar-row'>
      <div class='bar-label'>{escape(label)}</div>
      <div class='bar-mid'>
        <div class='bar-note'>{fmt_num(rlo,1)}-{fmt_num(rhi,1)}{pct_suffix}{(' '+unit_suffix) if unit_suffix and not pct_suffix else ''}</div>
        <div class='track'><div class='ref-zone' style='left:{rp1:.1f}%;width:{max(1,rp2-rp1):.1f}%'></div><span class='marker' style='left:{vp:.1f}%'></span></div>
      </div>
      <div class='bar-value'>{fmt_num(value,1)}{pct_suffix}{(' '+unit_suffix) if unit_suffix and not pct_suffix else ''}</div>
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
    if metric == "muscle":
        return seg.muscle_kg
    if metric == "ffm":
        return seg.ffm_kg
    return seg.fat_kg


def _segment_total(segments: Dict[str, SegmentValue], metric: str) -> Optional[float]:
    values = [_segment_metric_value(v, metric) for v in segments.values()]
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return sum(values) if values else None


def _segment_body_svg(data: ReportData, metric: str) -> str:
    """Continuous front-facing silhouette based on the user's clean reference image.

    The figure is intentionally one continuous body shape rather than assembled torso/
    limb pieces. Regional values are communicated by the surrounding labels; colour
    identifies the panel (muscle vs fat), not a normal/high/low clinical rating.
    """
    female = str(data.profile.sex).lower().startswith("f")
    path_d = 'M 104.5 8.0 L 96.8 9.3 L 90.3 14.8 L 87.1 26.7 L 87.7 37.7 L 90.3 45.5 L 90.3 53.9 L 85.8 65.5 L 91.6 74.6 L 82.5 76.5 L 73.5 78.5 L 68.3 83.0 L 65.1 87.5 L 62.5 105.0 L 60.5 125.7 L 57.3 152.1 L 54.7 174.1 L 52.1 198.7 L 52.1 214.9 L 54.1 222.6 L 61.8 227.8 L 59.9 222.6 L 56.7 214.9 L 59.9 210.3 L 63.1 218.1 L 62.5 208.4 L 61.2 193.5 L 65.1 174.1 L 72.2 143.7 L 78.6 113.4 L 82.5 128.9 L 87.1 146.3 L 84.5 154.1 L 74.8 181.3 L 74.1 194.8 L 73.5 209.7 L 77.4 235.6 L 82.5 265.3 L 85.8 284.0 L 84.5 303.4 L 83.2 320.2 L 87.1 345.5 L 90.3 368.7 L 85.1 377.1 L 75.4 388.8 L 79.9 390.1 L 90.3 391.4 L 97.4 386.8 L 101.9 382.3 L 102.5 367.4 L 103.9 335.8 L 105.8 281.4 L 107.1 207.8 L 111.0 206.5 L 113.6 242.0 L 116.1 294.4 L 119.4 349.3 L 118.7 365.5 L 117.4 381.7 L 123.2 386.2 L 132.9 391.4 L 139.4 390.7 L 143.9 389.4 L 137.5 379.7 L 129.1 368.1 L 132.9 343.5 L 136.8 313.1 L 135.5 297.6 L 133.6 284.7 L 138.1 257.6 L 145.9 212.9 L 145.9 195.5 L 145.2 183.2 L 140.7 169.6 L 132.9 148.9 L 137.5 128.9 L 141.4 113.4 L 147.8 143.7 L 154.9 174.1 L 158.2 192.2 L 157.6 207.8 L 156.2 218.1 L 159.5 210.3 L 162.7 213.6 L 160.1 221.3 L 156.9 227.2 L 162.7 224.6 L 165.3 223.3 L 167.2 211.0 L 165.3 190.9 L 163.3 169.6 L 160.8 143.1 L 158.8 119.2 L 156.9 100.5 L 154.9 88.8 L 151.7 83.6 L 147.8 79.1 L 137.5 76.5 L 126.5 73.9 L 130.4 65.5 L 132.9 60.4 L 131.0 47.4 L 129.7 34.5 L 128.4 20.3 L 124.5 14.8 L 120.0 10.6 L 112.9 8.7 Z' if female else 'M 107.5 8.0 L 101.9 9.2 L 98.2 11.7 L 94.5 15.4 L 91.4 21.0 L 90.2 30.9 L 89.6 40.2 L 93.9 47.0 L 97.0 51.4 L 97.0 63.1 L 88.9 67.4 L 77.2 72.4 L 64.2 78.6 L 59.2 84.8 L 56.1 92.2 L 53.6 106.5 L 51.8 119.5 L 49.9 134.4 L 46.8 155.4 L 46.8 176.5 L 46.2 198.1 L 46.2 218.0 L 49.9 225.4 L 53.6 231.6 L 58.0 232.8 L 62.9 232.8 L 59.2 227.3 L 55.5 222.3 L 56.1 217.4 L 58.0 214.2 L 61.1 217.4 L 64.2 221.7 L 62.3 213.6 L 57.4 192.6 L 60.5 180.2 L 64.2 162.8 L 66.0 151.1 L 66.0 143.0 L 69.7 134.4 L 74.1 125.7 L 77.8 139.3 L 81.5 153.5 L 78.4 172.1 L 75.3 192.6 L 74.7 212.4 L 72.8 243.4 L 75.9 265.0 L 73.5 283.0 L 71.0 303.4 L 73.5 326.9 L 76.6 353.6 L 78.4 367.8 L 72.2 377.7 L 63.5 389.5 L 71.0 390.7 L 78.4 391.4 L 85.8 386.4 L 90.8 380.9 L 90.8 370.3 L 88.9 356.1 L 92.0 341.2 L 97.6 317.7 L 96.4 291.7 L 99.5 269.9 L 103.8 239.7 L 106.9 218.6 L 111.2 216.1 L 115.5 244.6 L 119.9 272.4 L 123.6 294.1 L 121.8 314.6 L 125.5 333.1 L 131.1 357.9 L 129.2 381.5 L 136.0 386.4 L 145.3 391.4 L 151.5 390.7 L 156.5 388.9 L 149.7 379.0 L 141.6 369.1 L 143.5 352.4 L 147.2 332.5 L 149.0 317.1 L 146.6 295.4 L 144.1 266.9 L 147.2 241.5 L 146.6 224.8 L 144.7 195.0 L 142.2 174.0 L 138.5 157.9 L 142.2 141.2 L 145.3 126.3 L 149.7 133.1 L 152.7 139.3 L 153.4 151.1 L 154.0 164.1 L 158.3 177.7 L 162.0 189.5 L 160.8 199.4 L 155.2 221.1 L 159.5 216.1 L 162.0 214.2 L 163.9 217.4 L 164.5 221.7 L 161.4 226.0 L 157.1 231.0 L 158.9 234.1 L 164.5 232.8 L 167.0 231.0 L 170.7 225.4 L 173.2 220.4 L 173.2 200.0 L 172.6 177.1 L 172.6 151.1 L 170.1 134.4 L 168.2 119.5 L 166.4 106.5 L 162.6 88.5 L 158.9 82.3 L 154.0 77.4 L 143.5 72.4 L 132.9 67.4 L 123.0 63.7 L 122.4 52.6 L 126.1 47.0 L 129.8 41.4 L 129.2 31.5 L 128.0 19.8 L 124.2 14.2 L 119.9 10.5 L 114.9 8.6 Z'
    if metric == "muscle":
        c1, c2, edge = "#0f8fa9", "#0a6f8b", "#075d78"
    else:
        c1, c2, edge = "#d8a12f", "#b97822", "#9a641d"
    uid = f"{metric}-{'f' if female else 'm'}"
    return f"""
    <svg viewBox='0 0 220 400' class='segment-body-svg' role='img' aria-label='{metric.title()} regional body silhouette'>
      <path d='{path_d}' fill='{c1}' stroke='{edge}' stroke-width='1.05'
            stroke-linejoin='round'/>
    </svg>"""


def _segment_change(data: ReportData, position: str, metric: str) -> Optional[float]:
    previous = (data.diagnostics or {}).get("previous_segments")
    if not previous or position not in previous or position not in data.segments:
        return None
    cur = _segment_metric_value(data.segments[position], metric)
    prev = _segment_metric_value(previous[position], metric)
    if cur is None or prev is None:
        return None
    return float(cur) - float(prev)


def _segment_delta_html(data: ReportData, delta: Optional[float], metric: str) -> str:
    if delta is None:
        return "<span class='seg-delta neutral-delta'>no prior</span>"
    if abs(delta) < 0.05:
        return f"<span class='seg-delta neutral-delta'>→ {fmt_mass(0.0, data.profile, 1)}</span>"
    arrow = "↑" if delta > 0 else "↓"
    favourable = (metric == "muscle" and delta > 0) or (metric == "fat" and delta < 0)
    cls = "favourable" if favourable else "attention"
    return f"<span class='seg-delta {cls}'>{arrow} {fmt_mass(abs(delta), data.profile, 1)}</span>"


def _segment_label(data: ReportData, position: str, value: Optional[float], metric: str, side: str = "left") -> str:
    delta = _segment_change(data, position, metric)
    ffm = data.segments.get(position, SegmentValue()).ffm_kg
    ffm_line = f"<small class='seg-ffm'>FFM {fmt_mass(ffm, data.profile, 1)}</small>" if metric == "muscle" and ffm is not None else ""
    return f"""<div class='seg-readout {side}'>
      <span class='seg-name'>{escape(position.upper())}</span>
      <b>{fmt_mass(value, data.profile, 1)}</b>
      {ffm_line}
      {_segment_delta_html(data, delta, metric)}
    </div>"""


def _largest_segment_change(data: ReportData, metric: str, favourable_only: bool = True) -> Tuple[Optional[str], Optional[float]]:
    vals = []
    for pos in ["Left Arm", "Right Arm", "Torso", "Left Leg", "Right Leg"]:
        d = _segment_change(data, pos, metric)
        if d is not None:
            vals.append((pos, d))
    if not vals:
        return None, None
    if metric == "muscle":
        candidates = [(p,d) for p,d in vals if d > 0] if favourable_only else vals
        if not candidates:
            return None, None
        return max(candidates, key=lambda x: x[1])
    candidates = [(p,d) for p,d in vals if d < 0] if favourable_only else vals
    if not candidates:
        return None, None
    return min(candidates, key=lambda x: x[1])


def _segment_analysis_panel(data: ReportData, metric: str) -> str:
    seg = data.segments
    la, ra = seg.get("Left Arm", SegmentValue()), seg.get("Right Arm", SegmentValue())
    ll, rl = seg.get("Left Leg", SegmentValue()), seg.get("Right Leg", SegmentValue())
    torso = seg.get("Torso", SegmentValue())
    getv = lambda x: _segment_metric_value(x, metric)
    prev_date_obj = (data.diagnostics or {}).get("previous_segment_date")
    prev_date = pd.Timestamp(prev_date_obj).strftime("%d %b %Y") if prev_date_obj is not None else None
    torso_delta = _segment_change(data, "Torso", metric)
    if metric == "muscle":
        title, subtitle, cls = "SEGMENTAL MUSCLE ANALYSIS", "Withings regional muscle mass", "muscle"
        foot = f"Arm L/R difference {fmt_num(_balance(getv(la), getv(ra)),1,'%')} · Leg L/R difference {fmt_num(_balance(getv(ll), getv(rl)),1,'%')}"
    else:
        title, subtitle, cls = "SEGMENTAL FAT ANALYSIS", "Withings regional fat mass", "fat"
        foot = "Regional fat mass; arrows show change from previous complete segmental scan"
    prior_note = f"vs {prev_date}" if prev_date else "no prior complete segmental scan"
    return f"""
      <div class='seg-panel {cls}'>
        <div class='seg-panel-head'><div><b>{title}</b><span>{subtitle}</span></div><small>{prior_note}</small></div>
        <div class='seg-panel-body'>
          <div class='seg-side seg-left'>
            {_segment_label(data, 'Left Arm', getv(la), metric, 'right')}
            {_segment_label(data, 'Left Leg', getv(ll), metric, 'right')}
          </div>
          <div class='seg-figure'>
            {_segment_body_svg(data, metric)}
            <div class='seg-torso-chip'><span>TORSO</span><b>{fmt_mass(getv(torso),data.profile,1)}</b>{f"<small class='seg-ffm'>FFM {fmt_mass(torso.ffm_kg,data.profile,1)}</small>" if metric=='muscle' and torso.ffm_kg is not None else ''}{_segment_delta_html(data,torso_delta,metric)}</div>
          </div>
          <div class='seg-side seg-right'>
            {_segment_label(data, 'Right Arm', getv(ra), metric, 'left')}
            {_segment_label(data, 'Right Leg', getv(rl), metric, 'left')}
          </div>
        </div>
        <div class='seg-panel-foot'>{foot}</div>
      </div>"""


def _segment_trend_summary(data: ReportData) -> str:
    prev_date_obj = (data.diagnostics or {}).get("previous_segment_date")
    if prev_date_obj is None:
        return "<div class='segment-trend-empty'>No previous complete segmental scan available for regional change.</div>"
    prev_date = pd.Timestamp(prev_date_obj).strftime("%d %b %Y")

    def largest(metric: str):
        vals=[]
        for pos in ["Left Arm", "Right Arm", "Torso", "Left Leg", "Right Leg"]:
            d=_segment_change(data,pos,metric)
            if d is not None:
                vals.append((pos,float(d)))
        return max(vals,key=lambda x:abs(x[1])) if vals else (None,None)

    mp, md = largest("muscle")
    fp, fd = largest("fat")
    def fmt_change(pos, delta):
        if pos is None or delta is None:
            return "No comparison"
        arrow = "↑" if delta > 0.05 else ("↓" if delta < -0.05 else "→")
        return f"{pos} {arrow} {fmt_mass(abs(delta), data.profile, 1)}"

    muscle = fmt_change(mp, md)
    fat = fmt_change(fp, fd)
    return f"""<div class='segment-trend-summary'>
      <div><b>Largest observed muscle change</b><span>{escape(muscle)}</span></div>
      <div><b>Largest observed fat change</b><span>{escape(fat)}</span></div>
      <div><b>Interpretation note</b><span>Confirm regional changes across repeated scans</span></div>
      <small>vs {prev_date}</small>
    </div>"""


def _history_track(values: List[float], color: str = "#1c3f52", width: int = 1000, height: int = 78) -> str:
    """Responsive InBody-like history line that uses the full horizontal plot area."""
    if not values:
        return f"<svg viewBox='0 0 {width} {height}' class='history-svg'></svg>"
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return f"<svg viewBox='0 0 {width} {height}' class='history-svg'></svg>"
    lo, hi = float(finite.min()), float(finite.max())
    if math.isclose(lo, hi):
        lo -= 1.0; hi += 1.0
    pad = max((hi-lo)*0.34, 0.30)
    lo -= pad; hi += pad
    n = len(arr)
    xpad = 42 if n <= 8 else 34
    xs = np.linspace(xpad, width-xpad, n) if n > 1 else np.array([width/2])
    def yy(v):
        return height - 13 - (v-lo)/(hi-lo)*(height-30)
    pts=[]; dots=[]; labels=[]; guides=[]
    for i,(x,v) in enumerate(zip(xs,arr)):
        y=yy(float(v)); pts.append(f"{x:.1f},{y:.1f}")
        guides.append(f"<line x1='{x:.1f}' x2='{x:.1f}' y1='11' y2='{height-11}' class='hguide'/>")
        if i == n-1:
            dots.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='8.0' class='hlatest-halo'/><circle cx='{x:.1f}' cy='{y:.1f}' r='5.0' fill='{color}' class='hlatest'/>")
            label_cls='hval hval-latest'
        else:
            dots.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4.2' fill='{color}'/>")
            label_cls='hval'
        labels.append(f"<text x='{x:.1f}' y='{max(11,y-8):.1f}' text-anchor='middle' class='{label_cls}'>{v:.1f}</text>")
    return f"""<svg viewBox='0 0 {width} {height}' class='history-svg'>
      {''.join(guides)}
      <line x1='10' x2='{width-10}' y1='{height-10}' y2='{height-10}' class='hbase'/>
      <polyline points='{' '.join(pts)}' fill='none' stroke='{color}' stroke-width='3.4' vector-effect='non-scaling-stroke' stroke-linecap='round' stroke-linejoin='round'/>
      {''.join(dots)}{''.join(labels)}
    </svg>"""


def _history_panel(data: ReportData) -> str:
    h = data.history.copy().sort_values("Date")
    if h.empty:
        return "<div class='history-empty'>No complete scan history.</div>"
    h = h.tail(int(max(3, min(16, data.profile.history_points or 8))))
    dates = [pd.Timestamp(d).strftime("%d %b") for d in h["Date"]]
    factor = LB_PER_KG if _is_us(data.profile) else 1.0
    mu = mass_unit(data.profile)
    weights = [float(v)*factor for v in h["Weight (kg)"]]
    muscle = [float(v)*factor for v in h["Muscle mass (kg)"]]
    fat = [float(v) for v in h["Body fat %"]]
    day_span = max(0, (pd.Timestamp(h["Date"].iloc[-1]).floor("D") - pd.Timestamp(h["Date"].iloc[0]).floor("D")).days)

    def latest_monthly(col, suffix, convert_mass=False):
        vals = [float(v) for v in data.monthly[col].tolist() if not pd.isna(v)]
        if not vals:
            return "-"
        v = vals[-1] * (factor if convert_mass else 1.0)
        return f"{v:.1f}{suffix}"

    def delta_text(vals, suffix):
        if len(vals) < 2:
            return "-"
        d = vals[-1] - vals[0]
        arrow = "↑" if d > 0 else ("↓" if d < 0 else "→")
        return f"{arrow} {abs(d):.1f}{suffix}"

    def row(label, sublabel, vals, median_text, suffix):
        return f"""<div class='ih-row'>
          <div class='ih-label'>
            <b>{label}</b><span>{sublabel}</span>
            <small>Monthly median<br><strong>{median_text}</strong></small>
          </div>
          <div class='ih-plot'>{_history_track(vals, width=1000, height=78)}</div>
          <div class='ih-change'><span>{day_span}-DAY Δ</span><b>{delta_text(vals, suffix)}</b><small>{dates[0]} → {dates[-1]}</small></div>
        </div>"""

    n=len(dates)
    date_parts=[]
    for i,d in enumerate(dates):
        show = n <= 8 or i % 2 == 0 or i == n-1
        day,mon=d.split()
        date_parts.append(f"<span class='{'date-muted' if not show else ''}'>{f'<b>{day}</b><small>{mon}</small>' if show else ''}</span>")
    date_cells = "".join(date_parts)
    return f"""<div class='ih-wrap'>
        {row('Weight',mu,weights,latest_monthly('Weight (kg)',f' {mu}',True),f' {mu}')}
        {row('Muscle Mass',mu,muscle,latest_monthly('Muscle mass (kg)',f' {mu}',True),f' {mu}')}
        {row('Body Fat','%',fat,latest_monthly('Body fat %','%'),' pp')}
        <div class='ih-dates'><div></div><div class='ih-date-grid' style='grid-template-columns:repeat({len(dates)},1fr)'>{date_cells}</div><div></div></div>
        <div class='ih-note'>{escape(data.profile.history_daily_rule)} per measured day · {len(dates)} days shown · latest point emphasized · monthly medians use all complete scans and remain gap-aware.</div>
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
    profile_id = (data.profile.profile_id or "").strip()
    mu = mass_unit(data.profile)
    mass_factor = LB_PER_KG if _is_us(data.profile) else 1.0
    height_text = fmt_height(data.profile.height_m, data.profile)

    bf_status, bf_cls = _status(data.body_fat_pct, data.bands.body_fat)
    mm_status, mm_cls = _status(data.muscle_pct, data.bands.muscle_pct)
    water_status, water_cls = _status(data.water_pct, data.bands.water_pct)
    bmi_status, bmi_cls = _bmi_status(data.bmi, data.bands.bmi)
    vis_status, vis_cls = _visceral_status(data.visceral_fat, data.bands.visceral_fat)
    weight_cls = bmi_cls

    # Slightly more practitioner-friendly wording while keeping the underlying
    # reference logic unchanged.
    def display_status(label: str, cls: str, kind: str = "") -> str:
        mapping = {
            "Within ref.": "Within reference",
            "Above ref.": "Above reference",
            "Below ref.": "Below reference",
            "Just above": "Just above reference",
            "Just below": "Just below reference",
            "Above normal": "Above BMI reference" if kind == "bmi" else "Above reference",
            "Below normal": "Below BMI reference" if kind == "bmi" else "Below reference",
        }
        return mapping.get(label, label)

    bf_status_d = display_status(bf_status, bf_cls)
    mm_status_d = display_status(mm_status, mm_cls)
    water_status_d = display_status(water_status, water_cls)
    bmi_status_d = display_status(bmi_status, bmi_cls, "bmi")
    vis_status_d = display_status(vis_status, vis_cls)

    bmi_lo_w = data.bands.bmi[0] * data.profile.height_m ** 2
    bmi_hi_w = data.bands.bmi[1] * data.profile.height_m ** 2

    scan_date = data.scan_date.strftime("%d %b %Y")
    scan_time = data.scan_date.strftime("%H:%M")
    seg_date = data.segment_date.strftime("%d %b %Y") if data.segment_date is not None else "No segmental scan"

    vis_value = fmt_num(data.visceral_fat, 1)
    bmr_value = f"{data.bmr:,.0f} kcal/day" if data.bmr is not None else "-"
    meta_age = f"{data.metabolic_age:.0f} y" if data.metabolic_age is not None else "-"
    vasc_age = f"{data.vascular_age:.0f} y" if data.vascular_age is not None else "-"
    icw = fmt_mass(data.icw_kg, data.profile, 0)
    ecw = fmt_mass(data.ecw_kg, data.profile, 0)
    ecw_tbw = fmt_num(data.ecw_tbw_pct, 1, "%*")

    diag = data.diagnostics or {}

    def signed(value: Optional[float], decimals: int = 1, suffix: str = "") -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "-"
        if abs(float(value)) < 0.00001:
            return f"→ 0.{''.join(['0']*decimals)}{suffix}" if decimals else f"→ 0{suffix}"
        arrow = "↑" if value > 0 else "↓"
        return f"{arrow} {abs(value):.{decimals}f}{suffix}"

    prev_w = diag.get("previous_weight_kg")
    prev_fm = diag.get("previous_fat_mass_kg")
    prev_mm = diag.get("previous_muscle_mass_kg")
    prev_bf = diag.get("previous_body_fat_pct")
    d_w = data.weight_kg - prev_w if prev_w is not None else None
    d_fm = data.fat_mass_kg - prev_fm if prev_fm is not None else None
    d_mm = data.muscle_mass_kg - prev_mm if prev_mm is not None else None
    d_bf = data.body_fat_pct - prev_bf if prev_bf is not None else None
    prev_date_obj = diag.get("previous_scan_date")
    prev_date = pd.Timestamp(prev_date_obj).strftime("%d %b %Y") if prev_date_obj is not None else "-"

    med_w = diag.get("median30_weight_kg")
    med_fm = diag.get("median30_fat_mass_kg")
    med_mm = diag.get("median30_muscle_mass_kg")
    med_bf = diag.get("median30_body_fat_pct")
    m_w = data.weight_kg - med_w if med_w is not None else None
    m_fm = data.fat_mass_kg - med_fm if med_fm is not None else None
    m_mm = data.muscle_mass_kg - med_mm if med_mm is not None else None
    m_bf = data.body_fat_pct - med_bf if med_bf is not None else None

    prev_vis = diag.get("previous_visceral_fat")
    d_vis = data.visceral_fat - prev_vis if (data.visceral_fat is not None and prev_vis is not None) else None

    # Longitudinal trend diagnostics
    t30 = diag.get("trend30") or {}
    t90 = diag.get("trend90") or {}
    t30_w = diag.get("trend30_weight_delta")
    t30_fm = diag.get("trend30_fat_mass_delta")
    t30_mm = diag.get("trend30_muscle_delta")
    t30_bf = diag.get("trend30_body_fat_delta")
    t90_w = diag.get("trend90_weight_delta")
    t90_fm = diag.get("trend90_fat_mass_delta")
    t90_mm = diag.get("trend90_muscle_delta")
    t90_bf = diag.get("trend90_body_fat_delta")
    consistency = diag.get("measurement_consistency") or {}
    scan_quality = diag.get("scan_quality") or {}

    goal = data.profile.goal_weight_kg
    target_bf = data.profile.target_body_fat_pct
    start_weight = data.profile.starting_weight_kg
    start_date = data.profile.journey_start_date

    if goal is not None:
        goal_gap = data.weight_kg - goal
        projected_fat = goal - data.fat_free_mass_kg
        projected_pct = projected_fat / goal * 100 if goal > 0 else None
        projection_valid = projected_fat >= 0
        goal_summary = f"{fmt_mass(abs(goal_gap), data.profile, 2)} remaining"

        progress_html = ""
        if start_weight is not None and not math.isclose(float(start_weight), float(goal)):
            start_w = float(start_weight)
            goal_w = float(goal)
            direction = 1.0 if goal_w > start_w else -1.0
            total_distance = abs(goal_w - start_w)
            achieved = (data.weight_kg - start_w) * direction
            raw_progress = (achieved / total_distance * 100.0) if total_distance else 0.0
            shown_progress = max(0.0, min(100.0, raw_progress))
            start_label = f"Start {fmt_mass(start_w, data.profile, 1)}"
            if start_date is not None:
                start_label += f" · {start_date.strftime('%d %b %Y')}"
            if achieved >= 0:
                achieved_label = f"{fmt_mass(min(max(0.0, achieved), total_distance),data.profile,1)} toward goal"
            else:
                achieved_label = f"{fmt_mass(abs(achieved),data.profile,1)} away from goal"
            remaining_distance = max(0.0, (goal_w - data.weight_kg) * direction)
            progress_html = f"""
              <div class='journey-row'><span>{escape(start_label)}</span><b>{raw_progress:.0f}% complete</b></div>
              <div class='journey-track'><i style='width:{shown_progress:.1f}%'></i></div>
              <div class='journey-foot'><span>{escape(achieved_label)}</span><span>{fmt_mass(remaining_distance,data.profile,1)} remaining</span></div>
            """
        else:
            progress_html = f"""
              <div class='goal-route'>
                <div class='goal-route-end'><span>Current</span><b>{fmt_mass(data.weight_kg,data.profile,1)}</b></div>
                <div class='goal-route-track'><i>Current → Goal</i></div>
                <div class='goal-route-end goal'><span>Goal</span><b>{fmt_mass(goal,data.profile,1)}</b></div>
              </div>
              <div class='goal-route-note'>Add a starting weight to unlock journey progress %.</div>
            """

        goal_projection = (
            f"<div class='goal-proj'>Projected body fat at weight goal <b>{projected_pct:.1f}%</b></div>"
            if projection_valid else "<div class='goal-proj warn-text'>Weight goal is below current fat-free mass</div>"
        )
    else:
        threshold = data.bands.bmi[1] * data.profile.height_m ** 2
        goal_summary = "No weight goal"
        progress_html = f"""
          <div class='goal-distance'>
            <div class='goal-end'><b>{display_mass_value(threshold,data.profile):.1f}</b><span>BMI {data.bands.bmi[1]:.1f} {mu}</span></div>
            <div class='goal-arrow'><span></span><i>Reference boundary</i></div>
            <div class='goal-end current'><b>{display_mass_value(data.weight_kg,data.profile):.1f}</b><span>Current {mu}</span></div>
          </div>
        """
        goal_projection = "<div class='goal-proj'>Reference boundary shown; not a personal target.</div>"

    target_bf_html = ""
    if target_bf is not None:
        if 0 < float(target_bf) < 100:
            target_bf_html = f"<div class='target-bf'><span>Body-fat goal</span><b>{float(target_bf):.1f}%</b></div>"

    # Hide unavailable scan parameters rather than printing dash placeholders.
    param_items = []
    if data.bmr is not None: param_items.append(("BMR", f"{data.bmr:,.0f} kcal/day"))
    if data.metabolic_age is not None: param_items.append(("Metabolic age", f"{data.metabolic_age:.0f} y"))
    if data.vascular_age is not None: param_items.append(("Vascular age", f"{data.vascular_age:.0f} y"))
    if data.icw_kg is not None and data.ecw_kg is not None: param_items.append(("ICW / ECW", f"{display_mass_value(data.icw_kg,data.profile):.0f} / {display_mass_value(data.ecw_kg,data.profile):.0f} {mu}"))
    if data.ecw_tbw_pct is not None: param_items.append(("ECW/TBW", f"{data.ecw_tbw_pct:.1f}%*"))
    if data.visceral_fat is not None: param_items.append(("Visceral fat", f"{data.visceral_fat:.1f} / 20"))
    if data.pwv_mps is not None: param_items.append(("Pulse wave velocity", f"{data.pwv_mps:.1f} m/s"))
    if data.scan_heart_rate_bpm is not None: param_items.append(("Scan heart rate", f"{data.scan_heart_rate_bpm:.0f} bpm"))
    params_html = "".join(f"<div class='param-row'><span>{escape(k)}</span><b>{escape(v)}</b></div>" for k,v in param_items)
    if not params_html:
        params_html = "<div class='mini-note'>No additional Body Scan parameters were available in this export.</div>"

    id_line = f"<span class='patient-id'>ID {escape(profile_id)}</span>" if profile_id else ""

    kpi_delta_weight = fmt_mass_delta(d_w, data.profile, 2) if d_w is not None else "No prior day"
    kpi_delta_bf = signed(d_bf, 1, " pp") if d_bf is not None else "No prior day"
    kpi_delta_mm = fmt_mass_delta(d_mm, data.profile, 2) if d_mm is not None else "No prior day"
    kpi_delta_vis = signed(d_vis, 1, "") if d_vis is not None else "Ref 0–5"

    snapshot_prev_value = fmt_mass_delta(d_w,data.profile,2) if d_w is not None else "No prior day"
    snapshot_prev_sub = f"Body fat {signed(d_bf,1,' pp')} · {prev_date}" if d_w is not None else "Need another measured day"

    def period_snapshot(trend, dw, dbf, dmm):
        if not trend or dw is None:
            return "Insufficient history", "No suitable baseline scan"
        span = int(trend.get("span_days", 0))
        return fmt_mass_delta(dw,data.profile,1), f"BF {signed(dbf,1,' pp')} · Muscle {fmt_mass_delta(dmm,data.profile,1)} · {span}d span"

    snapshot_30_value, snapshot_30_sub = period_snapshot(t30, t30_w, t30_bf, t30_mm)
    snapshot_90_value, snapshot_90_sub = period_snapshot(t90, t90_w, t90_bf, t90_mm)
    snapshot_goal_value = goal_summary if goal is not None else f"BMI {data.bands.bmi[1]:.1f} boundary"
    if target_bf is not None:
        snapshot_goal_sub = f"Body-fat goal {float(target_bf):.1f}%"
    elif start_weight is not None and goal is not None:
        snapshot_goal_sub = f"Started at {fmt_mass(float(start_weight),data.profile,1)}"
    else:
        snapshot_goal_sub = "Profile-defined goal" if goal is not None else "Reference boundary only"

    def trend_rows(trend, dw, dbf, dfm, dmm):
        if not trend or dw is None:
            return "<div class='trend-empty'>Insufficient history for this interval.</div>"
        rows = [
            ("Weight", fmt_mass_delta(dw,data.profile,2)),
            ("Body fat", signed(dbf,1," pp")),
            ("Fat mass", fmt_mass_delta(dfm,data.profile,2)),
            ("Muscle mass", fmt_mass_delta(dmm,data.profile,2)),
        ]
        return "".join(f"<div class='trend-row'><span>{k}</span><b>{v}</b></div>" for k,v in rows)

    trend30_html = trend_rows(t30,t30_w,t30_bf,t30_fm,t30_mm)
    trend90_html = trend_rows(t90,t90_w,t90_bf,t90_fm,t90_mm)
    trend30_span = f"{int(t30.get('span_days',0))}d span" if t30 else "No suitable baseline"
    trend90_span = f"{int(t90.get('span_days',0))}d span" if t90 else "No suitable baseline"
    rate = diag.get("weight_rate_per_week")
    rate_text = (fmt_mass_delta(rate,data.profile,2) + "/week") if rate is not None else "Insufficient history"
    consistency_label = escape(str(consistency.get("label","No data")))
    quality_label = escape(str(scan_quality.get("label","No data")))
    quality_cls = str(scan_quality.get("class","neutral"))
    consistency_cls = str(consistency.get("class","neutral"))
    consistency_med = consistency.get("median_deviation_hours")
    consistency_max = consistency.get("max_deviation_hours")
    if consistency_med is not None and consistency_max is not None:
        consistency_detail = f"Median timing deviation {float(consistency_med):.1f} h · max {float(consistency_max):.1f} h · {int(consistency.get('scan_count',0))} recent scans"
    else:
        consistency_detail = f"{int(consistency.get('scan_count',0))} recent scans"

    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><style>
@page {{ size:A4 portrait; margin:0; }}
* {{ box-sizing:border-box; }}
html,body {{ margin:0; padding:0; background:#eef3f6; font-family:Arial,Helvetica,sans-serif; color:#164864; }}
.report-page {{ width:210mm; height:297mm; margin:0 auto; background:#fff; position:relative; overflow:hidden; }}
.header {{ height:23mm; background:#095c86; color:#fff; padding:3.8mm 6.2mm 3.2mm; display:grid; grid-template-columns:1.08fr 1fr; align-items:center; }}
.header h1 {{ margin:0; font-size:17pt; line-height:1; letter-spacing:.2px; }}
.header .subtitle {{ margin-top:1.5mm; font-size:6.8pt; opacity:.94; }}
.patient {{ text-align:right; }}
.patient-name {{ font-size:11.2pt; font-weight:800; letter-spacing:.15px; line-height:1.1; }}
.patient-meta {{ margin-top:1.1mm; font-size:6.8pt; line-height:1.35; }}
.patient-id {{ display:inline-block; margin-right:1.5mm; padding:.35mm 1.2mm; border:1px solid rgba(255,255,255,.5); border-radius:4mm; font-size:5.7pt; }}
.scan-meta {{ margin-top:.7mm; font-size:6pt; opacity:.9; }}
.content {{ padding:2.8mm 5.7mm 4mm; }}

.card {{ border:1px solid #b8d0df; border-radius:2mm; overflow:hidden; background:#fff; }}
.section-title {{ background:#086c9a; color:#fff; height:5.8mm; padding:1.15mm 2.5mm; font-size:7.2pt; font-weight:800; letter-spacing:.2px; }}
.section-title.split {{ display:flex; justify-content:space-between; align-items:center; }}
.section-title small {{ font-size:5.3pt; font-weight:600; }}

.kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:2.1mm; margin-bottom:2mm; }}
.kpi {{ height:17.5mm; border:1px solid #b8d0df; border-radius:2.2mm; padding:1.8mm 2.2mm; position:relative; }}
.kpi-label {{ font-size:5.8pt; font-weight:800; color:#647b8b; }}
.kpi-value {{ font-size:14.3pt; font-weight:800; color:#005b8e; margin-top:.55mm; line-height:1; }}
.kpi-status {{ position:absolute; top:7.3mm; right:2.3mm; width:3.2mm; height:3.2mm; border-radius:50%; }}
.kpi-foot {{ position:absolute; bottom:1.25mm; left:2.2mm; right:2.2mm; display:flex; justify-content:space-between; align-items:center; font-size:4.8pt; color:#6e818c; }}
.kpi-delta {{ font-weight:800; color:#285a75; }}
.good {{ background:#4db66a; }} .warn {{ background:#efad2e; }} .bad {{ background:#d65c51; }} .neutral {{ background:#9aaab4; }}

.snapshot {{ display:grid; grid-template-columns:repeat(4,1fr); border:1px solid #c7d9e3; border-radius:2mm; margin-bottom:2mm; overflow:hidden; background:#f7fbfd; }}
.snap-item {{ min-height:11.7mm; padding:1.4mm 2.2mm; border-right:1px solid #dce7ed; }}
.snap-item:last-child {{ border-right:0; }}
.snap-label {{ font-size:4.9pt; font-weight:800; color:#738794; text-transform:uppercase; letter-spacing:.2px; }}
.snap-value {{ margin-top:.7mm; font-size:8.0pt; font-weight:800; color:#0a5e8a; line-height:1.15; }}
.snap-sub {{ margin-top:.45mm; font-size:4.7pt; color:#6c7f89; }}

.two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:2.4mm; margin-bottom:2.2mm; }}
.comp-body,.bars {{ height:32.5mm; }}
.comp-body {{ padding:1mm 2.3mm; }}
.comp-row {{ display:grid; grid-template-columns:1fr 30mm 23mm; align-items:center; border-bottom:1px solid #edf1f4; height:5.05mm; font-size:6.05pt; }}
.comp-row:last-child {{ border-bottom:0; }}
.comp-row b {{ text-align:right; color:#005b8e; font-size:6.7pt; }}
.comp-row small {{ text-align:right; color:#7a8c96; font-size:5.1pt; }}
.bars {{ padding:.8mm 2.3mm; }}
.bar-row {{ display:grid; grid-template-columns:22mm 1fr 20mm; gap:1.7mm; align-items:center; height:6.0mm; font-size:6pt; }}
.bar-label {{ font-weight:700; }} .bar-value {{ text-align:right; font-weight:800; font-size:6.8pt; color:#005b8e; }}
.bar-note {{ font-size:4.7pt; color:#7b8d96; margin-bottom:.4mm; }}
.track {{ height:2.3mm; border-radius:2mm; background:#e5edf2; position:relative; }}
.ref-zone {{ position:absolute; top:0; bottom:0; background:#caead3; border-radius:2mm; }}
.marker {{ position:absolute; top:50%; width:3mm; height:3mm; margin-left:-1.5mm; margin-top:-1.5mm; border-radius:50%; background:#075b84; box-shadow:0 0 0 .4mm #fff; }}

.segment-card {{ margin-bottom:2.2mm; }}
.segment-wrap {{ padding:1.2mm 1.7mm 1.1mm; height:58.5mm; }}
.segment-panels {{ display:grid; grid-template-columns:1fr 1fr; gap:2.1mm; height:49.2mm; }}
.seg-panel {{ border:1px solid #cad9e2; background:#f7fafc; overflow:hidden; position:relative; }}
.seg-panel-head {{ height:7mm; padding:1.15mm 1.8mm; background:#eef5f8; display:flex; justify-content:space-between; align-items:flex-start; gap:1.5mm; }}
.seg-panel-head b {{ display:block; font-size:7pt; color:#086a97; }} .seg-panel.fat .seg-panel-head b {{ color:#9b680d; }}
.seg-panel-head span {{ display:block; font-size:4.55pt; color:#718591; margin-top:.15mm; }}
.seg-panel-head small {{ font-size:3.8pt; color:#80919a; white-space:nowrap; padding-top:.25mm; }}
.seg-panel-body {{ height:36.5mm; display:grid; grid-template-columns:1fr 27mm 1fr; align-items:center; padding:0 3.2mm; }}
.seg-side {{ height:100%; display:flex; flex-direction:column; justify-content:space-around; padding:4.4mm 0 2.3mm; }}
.seg-readout {{ font-size:5.0pt; color:#537080; line-height:1.08; }} .seg-readout.right {{ text-align:right; }} .seg-readout.left {{ text-align:left; }}
.seg-readout b {{ display:block; font-size:7.3pt; color:#00628f; margin-top:.25mm; }} .seg-panel.fat .seg-readout b {{ color:#a36b05; }} .seg-ffm {{display:block;font-size:3.45pt;color:#6b7f88;line-height:1.05;margin-top:.18mm;}}
.seg-name {{ font-size:4.55pt; font-weight:800; color:#607784; }}
.seg-delta {{ display:block; margin-top:.35mm; font-size:4.45pt; font-weight:800; letter-spacing:.02em; }}
.seg-delta.favourable {{ color:#2b8b55; }} .seg-delta.attention {{ color:#b56b24; }} .seg-delta.neutral-delta {{ color:#87959c; font-weight:600; }}
.seg-figure {{ position:relative; height:100%; display:flex; align-items:flex-end; justify-content:center; padding-bottom:.7mm; }}
.segment-body-svg {{ width:26.5mm; height:36.3mm; display:block; }}
.seg-torso-chip {{ position:absolute; left:50%; top:15.0mm; transform:translate(-50%,-50%); z-index:3; min-width:18mm; padding:.65mm 1mm .55mm; border-radius:2mm; background:rgba(255,255,255,.94); border:.25mm solid rgba(20,74,99,.16); box-shadow:0 .25mm .7mm rgba(25,58,74,.10); text-align:center; line-height:1.02; }}
.seg-torso-chip span {{ display:block; font-size:3.75pt; font-weight:800; color:#607784; }}
.seg-torso-chip b {{ display:block; margin-top:.15mm; font-size:6.2pt; color:#075c86; }} .seg-panel.fat .seg-torso-chip b {{ color:#8c5b0a; }}
.seg-torso-chip .seg-delta {{ font-size:3.55pt; margin-top:.25mm; }}
.seg-panel-foot {{ height:5.7mm; border-top:1px solid #dbe5ea; display:flex; align-items:center; justify-content:center; font-size:4.05pt; color:#6b7e88; padding:0 1mm; text-align:center; }}
.segment-trend-summary {{ height:6.7mm; margin-top:1mm; display:grid; grid-template-columns:.92fr .92fr 1.18fr auto; align-items:center; gap:2mm; padding:.7mm 1.5mm; border-radius:1.2mm; background:#edf5f8; color:#486878; font-size:4.25pt; }}
.segment-trend-summary div {{ display:flex; align-items:baseline; gap:1.2mm; min-width:0; }}
.segment-trend-summary b {{ color:#2d5b70; white-space:nowrap; }} .segment-trend-summary span {{ color:#075c86; font-weight:800; white-space:nowrap; }}
.segment-trend-summary small {{ color:#71848f; font-size:3.85pt; white-space:nowrap; }}
.segment-trend-empty {{ height:6.7mm; margin-top:1mm; display:flex; align-items:center; justify-content:center; border-radius:1.2mm; background:#f2f6f8; color:#7a8a93; font-size:4pt; }}

.history-card {{ margin-bottom:2.2mm; }}
.history-inner {{ padding:.8mm 1.1mm .7mm; height:54mm; }}
.ih-wrap {{ height:100%; }}
.ih-row {{ display:grid; grid-template-columns:20.5mm 1fr 23mm; height:13.1mm; border-bottom:1px solid #e6edf1; align-items:center; }}
.ih-label {{ height:100%; background:#edf4f7; padding:1.35mm 1.35mm; }}
.ih-label b {{ display:block; font-size:6.55pt; color:#294f62; line-height:1.05; }}
.ih-label span {{ display:block; font-size:4.45pt; color:#748791; }}
.ih-label small {{ display:block; margin-top:.65mm; font-size:3.85pt; color:#70838e; line-height:1.05; }} .ih-label small strong {{ color:#355c70; }}
.ih-plot {{ padding:.05mm .25mm; overflow:hidden; min-width:0; }}
.history-svg {{ width:100%; height:12.1mm; display:block; overflow:visible; }}
.history-svg .hbase {{ stroke:#d8e0e4; stroke-width:1; }} .history-svg .hguide {{ stroke:#edf2f4; stroke-width:1; }}
.history-svg .hval {{ fill:#2b3e48; font-size:11.5px; font-weight:700; font-family:Arial,Helvetica,sans-serif; }}
.history-svg .hval-latest {{ fill:#075c86; font-weight:800; font-size:12.5px; }}
.history-svg .hlatest-halo {{ fill:#d7edf6; opacity:.9; }}
.ih-change {{ padding:1mm 1.1mm; border-left:1px solid #e4ecef; height:100%; display:flex; flex-direction:column; justify-content:center; }}
.ih-change span {{ font-size:3.9pt; font-weight:800; color:#788a94; }} .ih-change b {{ font-size:6.7pt; color:#0a5d87; margin-top:.3mm; }} .ih-change small {{ font-size:3.55pt; color:#85939b; margin-top:.25mm; }}
.ih-dates {{ display:grid; grid-template-columns:20.5mm 1fr 23mm; min-height:5.7mm; }}
.ih-date-grid {{ display:grid; text-align:center; align-items:start; padding:.45mm .25mm 0; }}
.ih-date-grid span {{ white-space:nowrap; line-height:1; min-width:0; }} .ih-date-grid b {{ display:block; font-size:3.9pt; color:#536c78; }} .ih-date-grid small {{ display:block; font-size:3.4pt; color:#82919a; }}
.ih-date-grid .date-muted {{ opacity:0; }}
.ih-note {{ margin-top:.25mm; padding:.6mm 1mm; background:#eef4f7; font-size:3.85pt; color:#6c808b; }}

.bottom-grid {{ display:grid; grid-template-columns:.92fr .98fr 1.1fr; gap:2.2mm; }}
.bottom-card {{ height:58mm; }}
.params,.change-inner,.goal-inner {{ padding:1.25mm 2.1mm; font-size:5.35pt; }}
.param-row,.math-row,.change-row {{ display:flex; justify-content:space-between; gap:2mm; line-height:1.48; }}
.param-row b,.math-row b,.change-row b {{ color:#005b8e; text-align:right; }}
.subbox-title {{ background:#edf5f8; color:#0b668f; font-size:5.25pt; font-weight:800; padding:.8mm 1.1mm; margin:1mm -.7mm .6mm; border-radius:1.1mm; }}
.note-box {{ background:#eef3f6; border-radius:1.3mm; padding:1.1mm; margin-top:.9mm; font-size:4.15pt; color:#597180; line-height:1.3; }}
.mini-note {{ margin-top:.7mm; font-size:4.05pt; color:#70828d; line-height:1.25; }}

.change-group {{ margin-bottom:1mm; }}
.change-head {{ display:flex; justify-content:space-between; align-items:end; margin-bottom:.7mm; }}
.change-head b {{ font-size:5.2pt; color:#315a70; }} .change-head small {{ font-size:3.8pt; color:#81919a; }}
.change-row {{ height:4.2mm; align-items:center; border-bottom:1px solid #eef2f4; }}
.change-row:last-child {{ border-bottom:0; }}
.change-row .chg {{ font-weight:800; color:#0a5d87; }}
.change-rule {{ margin-top:.8mm; padding:.8mm 1mm; background:#f3f7f9; border-radius:1mm; color:#667b86; font-size:4pt; line-height:1.25; }}

.trend-head {{ display:flex; justify-content:space-between; align-items:end; margin-bottom:.45mm; }}
.trend-head b {{ font-size:5.25pt; color:#315a70; }} .trend-head small {{ font-size:3.9pt; color:#81919a; }}
.trend-row {{ height:3.65mm; display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #eef2f4; font-size:5.0pt; }}
.trend-row b {{ color:#0a5d87; }}
.trend-empty {{ padding:1mm 0; color:#82919a; font-size:4.35pt; }}
.quality-box {{ margin-top:.8mm; padding:.8mm 1mm; background:#f1f6f8; border-radius:1.1mm; }}
.quality-top {{ display:flex; justify-content:space-between; align-items:center; gap:1mm; }}
.quality-top b {{ font-size:5.15pt; color:#315a70; }}
.quality-pill {{ border-radius:4mm; padding:.35mm 1.1mm; font-weight:800; font-size:4.1pt; color:#31644a; background:#d8f0df; }}
.quality-pill.warn {{ color:#8e5a06; background:#ffebc3; }} .quality-pill.neutral {{ color:#64737d; background:#e8edef; }}
.quality-detail {{ margin-top:.45mm; color:#647985; font-size:4.0pt; line-height:1.25; }}
.rate-row {{ margin-top:.6mm; display:flex; justify-content:space-between; font-size:4.7pt; }} .rate-row b {{ color:#0a5d87; }}

.goal-top {{ display:flex; justify-content:space-between; align-items:flex-start; }}
.goal-number {{ font-size:14.5pt; color:#005b8e; font-weight:800; line-height:1; margin-top:.6mm; }}
.goal-remaining {{ text-align:right; font-size:5.2pt; color:#e17b1d; font-weight:800; padding-top:2mm; }}
.goal-route {{ display:grid; grid-template-columns:24mm 1fr 24mm; gap:1.8mm; align-items:center; margin:1.4mm 0 .7mm; }}
.goal-route-end {{ display:flex; flex-direction:column; gap:.15mm; }}
.goal-route-end.goal {{ text-align:right; align-items:flex-end; }}
.goal-route-end span {{ font-size:3.75pt; color:#748791; }}
.goal-route-end b {{ font-size:6.8pt; color:#0a5d87; }}
.goal-route-track {{ position:relative; height:5mm; text-align:center; }}
.goal-route-track:before {{ content:''; position:absolute; left:0; right:0; top:1.9mm; height:.8mm; background:#d7e5eb; border-radius:2mm; }}
.goal-route-track:after {{ content:''; position:absolute; right:0; top:1.25mm; border-top:1mm solid transparent; border-bottom:1mm solid transparent; border-left:1.7mm solid #0a6c99; }}
.goal-route-track i {{ position:relative; z-index:2; background:#fff; padding:0 .8mm; font-size:3.5pt; color:#71848f; font-style:normal; }}
.goal-route-note {{ margin-top:.2mm; font-size:3.65pt; color:#7b8c95; }}
.goal-distance {{ display:grid; grid-template-columns:20mm 1fr 20mm; gap:1.5mm; align-items:center; margin:1.4mm 0 1mm; }}
.goal-end {{ text-align:left; }} .goal-end.current {{ text-align:right; }}
.goal-end b {{ display:block; font-size:7pt; color:#0a5d87; }} .goal-end span {{ font-size:3.8pt; color:#748791; }}
.goal-arrow {{ position:relative; height:5mm; text-align:center; }}
.goal-arrow span {{ position:absolute; left:0; right:0; top:1.9mm; height:.8mm; background:#d7e5eb; border-radius:2mm; }}
.goal-arrow:after {{ content:''; position:absolute; left:0; top:1.25mm; border-top:1mm solid transparent; border-bottom:1mm solid transparent; border-right:1.7mm solid #0a6c99; }}
.goal-arrow i {{ position:relative; z-index:2; background:#fff; padding:0 .8mm; font-size:3.5pt; color:#71848f; font-style:normal; }}
.goal-proj {{ font-size:5.3pt; color:#0b5f88; margin:.55mm 0; }} .goal-proj b {{ font-size:6.2pt; }}
.journey-row {{ display:flex; justify-content:space-between; gap:2mm; margin-top:1.2mm; font-size:4.7pt; color:#617884; }} .journey-row b {{ color:#0a5d87; }}
.journey-track {{ height:2.2mm; border-radius:2mm; background:#e1ebf0; overflow:hidden; margin:.55mm 0; }} .journey-track i {{ display:block; height:100%; background:#0a6c99; border-radius:2mm; }}
.journey-foot {{ display:flex; justify-content:space-between; font-size:3.95pt; color:#7c8d96; }}
.target-bf {{ display:flex; justify-content:space-between; align-items:baseline; margin:.55mm 0; padding:.55mm .9mm; background:#eef5f8; border-radius:1mm; font-size:4.65pt; }} .target-bf b {{ color:#0a5d87; font-size:6.0pt; }}
.note {{ font-size:3.95pt; color:#667b86; line-height:1.2; }}
.assess {{ margin-top:.35mm; }}
.assess-row {{ display:grid; grid-template-columns:1fr 13mm 3.5mm 25mm; gap:1mm; align-items:center; height:3.25mm; font-size:4.6pt; }}
.assess-value {{ text-align:right; font-weight:800; color:#0a5d87; font-size:4.2pt; }}
.assess-dot {{ width:2.4mm; height:2.4mm; border-radius:50%; }}
.pill {{ border-radius:4mm; text-align:center; padding:.32mm .6mm; color:#31644a; background:#d8f0df; font-weight:700; font-size:3.85pt; }}
.pill.warn {{ color:#8e5a06; background:#ffebc3; }} .pill.bad {{ color:#873c36; background:#f5d0cd; }} .pill.neutral {{ color:#64737d; background:#e8edef; }}
.warn-text {{ color:#a45d18; }}

.footer {{ position:absolute; left:5.8mm; right:5.8mm; bottom:2.8mm; display:flex; justify-content:space-between; font-size:3.95pt; color:#70818a; }}
</style></head><body>
<div class='report-page'>
  <div class='header'>
    <div>
      <h1>BODY COMPOSITION REPORT</h1>
      <div class='subtitle'>Withings Body Scan</div>
    </div>
    <div class='patient'>
      <div class='patient-name'>{escape(name)}</div>
      <div class='patient-meta'>{id_line}{escape(sex)} · {age_text} years · {height_text}</div>
      <div class='scan-meta'>Scan {scan_date} · {scan_time} &nbsp; | &nbsp; Segmental {seg_date}</div>
    </div>
  </div>

  <div class='content'>
    <div class='kpis'>
      <div class='kpi'>
        <div class='kpi-label'>WEIGHT</div><div class='kpi-value'>{fmt_mass(data.weight_kg,data.profile,2)}</div><i class='kpi-status {_dot_class(weight_cls)}'></i>
        <div class='kpi-foot'><span class='kpi-delta'>{kpi_delta_weight}</span><span>BMI {data.bmi:.1f}</span></div>
      </div>
      <div class='kpi'>
        <div class='kpi-label'>BODY FAT</div><div class='kpi-value'>{data.body_fat_pct:.1f}%</div><i class='kpi-status {_dot_class(bf_cls)}'></i>
        <div class='kpi-foot'><span class='kpi-delta'>{kpi_delta_bf}</span><span>{fmt_mass(data.fat_mass_kg,data.profile,2)}</span></div>
      </div>
      <div class='kpi'>
        <div class='kpi-label'>MUSCLE MASS</div><div class='kpi-value'>{fmt_mass(data.muscle_mass_kg,data.profile,2)}</div><i class='kpi-status {_dot_class(mm_cls)}'></i>
        <div class='kpi-foot'><span class='kpi-delta'>{kpi_delta_mm}</span><span>{data.muscle_pct:.1f}%</span></div>
      </div>
      <div class='kpi'>
        <div class='kpi-label'>VISCERAL FAT</div><div class='kpi-value'>{vis_value}</div><i class='kpi-status {_dot_class(vis_cls)}'></i>
        <div class='kpi-foot'><span class='kpi-delta'>{kpi_delta_vis}</span><span>{vis_status_d}</span></div>
      </div>
    </div>

    <div class='snapshot'>
      <div class='snap-item'><div class='snap-label'>Previous measured day</div><div class='snap-value'>{snapshot_prev_value}</div><div class='snap-sub'>{snapshot_prev_sub}</div></div>
      <div class='snap-item'><div class='snap-label'>30-day trend</div><div class='snap-value'>{snapshot_30_value}</div><div class='snap-sub'>{snapshot_30_sub}</div></div>
      <div class='snap-item'><div class='snap-label'>90-day trend</div><div class='snap-value'>{snapshot_90_value}</div><div class='snap-sub'>{snapshot_90_sub}</div></div>
      <div class='snap-item'><div class='snap-label'>Goal position</div><div class='snap-value'>{snapshot_goal_value}</div><div class='snap-sub'>{snapshot_goal_sub}</div></div>
    </div>

    <div class='two-col'>
      <div class='card'><div class='section-title'>BODY COMPOSITION ANALYSIS</div><div class='comp-body'>
        <div class='comp-row'><span>Total Body Water</span><b>{fmt_mass(data.water_kg,data.profile,2)}</b><small>{data.water_pct:.1f}%</small></div>
        <div class='comp-row'><span>Fat-Free Mass</span><b>{fmt_mass(data.fat_free_mass_kg,data.profile,2)}</b><small>Weight − fat mass</small></div>
        <div class='comp-row'><span>Muscle Mass</span><b>{fmt_mass(data.muscle_mass_kg,data.profile,2)}</b><small>{data.muscle_pct:.1f}%</small></div>
        <div class='comp-row'><span>Bone Mass</span><b>{fmt_mass(data.bone_mass_kg,data.profile,2)}</b><small>{data.bone_pct:.1f}%</small></div>
        <div class='comp-row'><span>Body Fat Mass</span><b>{fmt_mass(data.fat_mass_kg,data.profile,2)}</b><small>{data.body_fat_pct:.1f}%</small></div>
        <div class='comp-row'><span>Weight</span><b>{fmt_mass(data.weight_kg,data.profile,2)}</b><small>BMI {data.bmi:.1f}</small></div>
      </div></div>
      <div class='card'><div class='section-title'>MUSCLE · FAT / OBESITY ANALYSIS</div><div class='bars'>
        {_bar(display_mass_value(data.weight_kg,data.profile),(display_mass_value(bmi_lo_w,data.profile),display_mass_value(bmi_hi_w,data.profile)),(display_mass_value(max(35,bmi_lo_w*0.65),data.profile),display_mass_value(bmi_hi_w*1.45,data.profile)),'Weight',mu)}
        {_bar(data.muscle_pct,data.bands.muscle_pct,(45,95),'Muscle %')}
        {_bar(data.body_fat_pct,data.bands.body_fat,(5,45),'Body fat %')}
        {_bar(data.bmi,data.bands.bmi,(14,36),'BMI')}
        {_bar(data.water_pct,data.bands.water_pct,(35,75),'Water %')}
      </div></div>
    </div>

    <div class='card segment-card'>
      <div class='section-title split'><span>SEGMENTAL ANALYSIS</span><small>Latest segmental scan: {seg_date}</small></div>
      <div class='segment-wrap'><div class='segment-panels'>
        {_segment_analysis_panel(data, 'muscle')}
        {_segment_analysis_panel(data, 'fat')}
      </div>{_segment_trend_summary(data)}</div>
    </div>

    <div class='card history-card'>
      <div class='section-title split'><span>BODY COMPOSITION HISTORY</span><small>Recent measured days · selected rule: {escape(data.profile.history_daily_rule)}</small></div>
      <div class='history-inner'>{_history_panel(data)}</div>
    </div>

    <div class='bottom-grid'>
      <div class='card bottom-card'><div class='section-title'>BODY SCAN PARAMETERS</div><div class='params'>
        {params_html}
        <div class='subbox-title'>REFERENCE BANDS</div>
        <div class='param-row'><span>Body fat</span><b>{data.bands.body_fat[0]:g}–{data.bands.body_fat[1]:g}%</b></div>
        <div class='param-row'><span>Muscle</span><b>{data.bands.muscle_pct[0]:g}–{data.bands.muscle_pct[1]:g}%</b></div>
        <div class='param-row'><span>Body water</span><b>{data.bands.water_pct[0]:g}–{data.bands.water_pct[1]:g}%</b></div>
        <div class='param-row'><span>BMI</span><b>{data.bands.bmi[0]:g}–{data.bands.bmi[1]:g}</b></div>
        <div class='note-box'><b>BIA measurement note</b><br>Compare measurements under similar time and hydration conditions. Short-term shifts may reflect BIA variability.</div>
      </div></div>

      <div class='card bottom-card'><div class='section-title'>TREND & DATA QUALITY</div><div class='change-inner'>
        <div class='trend-head'><b>30-DAY TREND</b><small>{trend30_span}</small></div>
        {trend30_html}
        <div class='trend-head' style='margin-top:.8mm'><b>90-DAY TREND</b><small>{trend90_span}</small></div>
        {trend90_html}
        <div class='quality-box'>
          <div class='quality-top'><b>Comparison quality</b><span class='quality-pill {quality_cls if quality_cls!='good' else ''}'>{quality_label}</span></div>
          <div class='quality-detail'>Measurement timing: {consistency_label} · {escape(consistency_detail)}</div>
          <div class='rate-row'><span>Average weight rate</span><b>{rate_text}</b></div>
        </div>
      </div></div>

      <div class='card bottom-card'><div class='section-title'>GOAL & INTERPRETATION</div><div class='goal-inner'>
        <div class='goal-top'><div><div class='snap-label'>PERSONAL GOAL</div><div class='goal-number'>{fmt_mass(goal,data.profile,1) if goal is not None else 'Not supplied'}</div></div><div class='goal-remaining'>{goal_summary}</div></div>
        {progress_html}
        {target_bf_html}
        {goal_projection}
        <div class='note'>Goal projections assume current fat-free mass is maintained and are mathematical, not prescribed targets.</div>
        <div class='subbox-title'>CURRENT ASSESSMENT</div>
        <div class='assess'>
          <div class='assess-row'><span>Body fat</span><span class='assess-value'>{data.body_fat_pct:.1f}%</span><i class='assess-dot {bf_cls}'></i><span class='pill {bf_cls if bf_cls!='good' else ''}'>{bf_status_d}</span></div>
          <div class='assess-row'><span>Visceral fat</span><span class='assess-value'>{vis_value}</span><i class='assess-dot {vis_cls}'></i><span class='pill {vis_cls if vis_cls!='good' else ''}'>{vis_status_d}</span></div>
          <div class='assess-row'><span>Muscle %</span><span class='assess-value'>{data.muscle_pct:.1f}%</span><i class='assess-dot {mm_cls}'></i><span class='pill {mm_cls if mm_cls!='good' else ''}'>{mm_status_d}</span></div>
          <div class='assess-row'><span>BMI</span><span class='assess-value'>{data.bmi:.1f}</span><i class='assess-dot {bmi_cls}'></i><span class='pill {bmi_cls if bmi_cls!='good' else ''}'>{bmi_status_d}</span></div>
          <div class='assess-row'><span>Water %</span><span class='assess-value'>{data.water_pct:.1f}%</span><i class='assess-dot {water_cls}'></i><span class='pill {water_cls if water_cls!='good' else ''}'>{water_status_d}</span></div>
        </div>
      </div></div>
    </div>
  </div>

  <div class='footer'>
    <span>Withings-derived report. Muscle Mass is not InBody Skeletal Muscle Mass (SMM). No proprietary InBody Score is calculated.</span>
    <span>Whole-body {scan_date} · Segmental {seg_date}</span>
  </div>
</div></body></html>"""
    return html


def html_to_pdf_bytes(html: str) -> bytes:
    try:
        from weasyprint import HTML
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PDF export requires the optional 'weasyprint' package.") from exc
    return HTML(string=html).write_pdf()
