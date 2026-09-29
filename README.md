# Body Composition Intelligence — Report Engine V8.0

A Streamlit app that imports a Withings export ZIP (recommended) or the core CSV files and produces a one-page body-composition report with HTML/PDF export.

V8.0 finishes the report engine around three goals:

1. **Universal Withings import normalization**
2. **Independent Metric / US Customary report units**
3. **Better use of Body Scan data and comparison quality**

## New in V8.0

### Universal source-unit normalization

The importer no longer assumes that `weight.csv` is metric.

It detects explicit mass units from the export headers and normalizes body-mass values internally to **kg**. Supported source labels include common kg/lb variants such as:

- `Weight (kg)` / `Weight (lb)` / `Weight (lbs)`
- Fat mass
- Bone mass
- Muscle mass
- Hydration / body-water mass

Mass-valued rows from `other.csv` are also normalized, including:

- segmental muscle mass
- segmental fat mass
- segmental fat-free mass
- ICW / ECW

If a non-standard export omits mass-unit labels, use **Advanced import settings → Source mass unit override** (`kg` or `lb`). Normal Withings exports should remain on **Auto-detect**.

### Import units and report units are independent

The report can display either:

- **Metric** — kg / cm
- **US Customary** — lb / ft + in

Examples that are supported:

- Metric ZIP → Metric report
- Metric ZIP → US report
- US ZIP → Metric report
- US ZIP → US report

All analytics remain in canonical metric units internally. Conversion happens only at import and presentation boundaries.

PWV stays in **m/s** and heart rate stays in **bpm** in both report-unit modes.

### New Body Scan metrics

V8.0 uses additional Withings export data when available:

- **Segmental Fat-Free Mass (FFM)** for left/right arms, torso and left/right legs
- **Pulse Wave Velocity (PWV)** from `pwv.csv`
- **Scan heart rate** from `bp.csv`, matched to the whole-body scan timestamp

Segmental FFM is shown as a secondary value inside the segmental muscle analysis so the one-page report remains readable.

### Comparison-quality logic

V8.0 adds an internal scan-quality layer that considers:

- complete whole-body composition fields
- complete segmental comparison scans
- plausible composition values
- recent measurement-time consistency
- unusually large short-term body-water shifts
- nearby BIA Error records for diagnostics without inventing undocumented code meanings

The report exposes a conservative **Comparison quality** cue such as Good / Variable / Limited data. It is a data-quality aid, not a medical accuracy claim.

### Central comparison behavior

- **Headline values:** latest complete whole-body scan
- **Previous measured day:** previous representative measured day using the selected history rule
- **Segmental values:** latest complete muscle/fat segmental snapshot
- **Segmental change:** previous fully complete segmental snapshot only
- **30-day / 90-day trends:** representative comparison nearest the requested interval, suppressed if insufficient history exists
- **Monthly medians:** all complete scans; gaps remain gaps and are never interpolated

## History controls

The profile supports:

- Earliest complete scan
- Latest complete scan
- Daily median

History display supports 6 / 8 / 10 / 12 measured days.

The recent-history rule affects the compact recent-history panel only. Monthly medians continue to use all complete scans.

## Goals and journey tracking

All goal fields are optional:

- goal weight
- target body-fat %
- journey start date
- starting weight

When start and goal weights are supplied, the report shows real journey progress. Goal body-fat projections remain mathematical projections that assume current fat-free mass is maintained; they are not prescribed targets.

## Important terminology: `pp`

The report intentionally keeps **pp**, meaning **percentage points**.

Example: body fat falling from 21.6% to 20.0% is a change of **-1.6 percentage points (-1.6 pp)**. It is not a 1.6% relative reduction, so replacing `pp` with `%` would change the mathematical meaning.

## Withings vs InBody terminology

This is a Withings-derived report. Withings **Muscle Mass** is not proprietary InBody Skeletal Muscle Mass (SMM). The app does not fabricate an InBody Score, SMM, protein, mineral, waist-hip ratio or other values not present in the Withings export.

## ZIP files used by V8

Required:

- `weight.csv`
- `other.csv`

Automatically used when present:

- `pwv.csv`
- `bp.csv`

For manual CSV mode, `pwv.csv` and `bp.csv` can be uploaded optionally.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the local Streamlit URL, normally `http://localhost:8501`.

## Streamlit Community Cloud

Keep these in the repository root:

- `requirements.txt` — Python packages, including WeasyPrint
- `packages.txt` — Debian system dependencies required by WeasyPrint

After committing dependency changes, reboot the Streamlit app if needed.

## GitHub package files

```text
app.py
report_engine.py
requirements.txt
packages.txt
README.md
test_report_engine.py
.gitignore
```

Preview files are included in the release ZIP for visual comparison but are not required by Streamlit.

## V8.0 acceptance coverage

The automated suite covers the existing V7 behavior plus V8-specific checks for:

- kg source normalization
- lb source normalization
- unitless source fallback via explicit override
- US-source → Metric report
- canonical Metric → US report
- no `kg` leakage in US report HTML
- segmental FFM
- PWV
- scan heart rate
- scan-quality diagnostics
- recent-history rules
- 6/8/10/12 history behavior
- 30/90-day trend logic
- segmental comparison logic

Current package status: **14 tests passing**.
