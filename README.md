# Body Composition Intelligence — V7

A Streamlit app that imports a Withings export (`weight.csv` + `other.csv`, directly or inside the original Withings ZIP) and produces a one-page body-composition report with PDF export.

## V7 focus

V7 keeps the V6.2 visual structure but improves longitudinal interpretation, data quality, goal tracking, and print readability while keeping the report generic for any user.

### New in V7

- Header subtitle simplified to **Withings Body Scan** only.
- 30-day and 90-day trend summaries using the profile-selected daily history rule.
- Actual comparison span is retained internally and shown in the report (for example, a 29-day available interval for the 30-day trend).
- Average weight-change rate in kg/week when enough longitudinal data is available.
- Measurement-time consistency indicator based on recent complete scans. This is a data-quality cue, not a clinical diagnosis.
- Optional journey start date + starting weight.
- Real goal progress percentage when a starting weight and goal weight are provided.
- Optional target body-fat percentage.
- Current Assessment now shows the actual current value beside each reference status.
- Unavailable Body Scan parameters are hidden instead of displaying a dash.
- Segmental changes are described as **observed** changes and include a reminder to confirm regional changes across repeated scans.
- Segmental typography and body scale refined for print readability.
- History remains responsive for 6 / 8 / 10 / 12 measured days.
- PDF export remains supported through WeasyPrint.

## Important terminology: `pp`

The report intentionally keeps **pp**, meaning **percentage points**.

Example: if body fat changes from 21.6% to 20.0%, the change is **-1.6 percentage points (-1.6 pp)**. It is not a 1.6% relative reduction. Replacing `pp` with `%` would therefore change the mathematical meaning.

## How the report chooses data

- **Headline values:** latest complete whole-body scan.
- **History:** one complete scan per measured day according to the selected profile rule:
  - Earliest complete scan
  - Latest complete scan
  - Daily median
- **12-month monthly medians:** use all complete scans and remain gap-aware; missing months are not interpolated.
- **Segmental values:** latest self-contained segmental snapshot.
- **Segmental change:** latest complete segmental snapshot vs the previous fully complete segmental snapshot.
- **30-day / 90-day trend:** compares the current scan with the available daily representative scan closest to the requested interval, but suppresses the comparison when history is too short to reasonably represent the interval.

## Measurement consistency

The report evaluates recent scan-time consistency from the time of day of recent complete scans. This is meant to help interpret BIA variability. Hydration, meals, exercise, and measurement timing can materially affect short-term BIA readings.

## Goal logic

All goal fields are optional.

- **Goal weight:** profile-defined target.
- **Journey start date + starting weight:** enables a real progress percentage.
- **Target body fat %:** shown as a separate user-defined composition goal.
- The projected body-fat value at goal weight assumes current fat-free mass is maintained. It is a mathematical projection, not a prescribed target.

## Withings vs InBody terminology

This is a Withings-derived report. Withings **Muscle Mass** is not proprietary InBody Skeletal Muscle Mass (SMM). The app does not fabricate an InBody Score, SMM, protein, mineral, waist-hip ratio, or other values that are not present in the Withings export.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the local Streamlit URL (normally `http://localhost:8501`).

## Streamlit Community Cloud

Keep both files in the repository root:

- `requirements.txt` — Python packages, including `weasyprint`
- `packages.txt` — Debian system packages required by WeasyPrint

After committing dependency changes, reboot the Streamlit app if necessary.

## Project files

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

## Tests

```bash
pytest -q
```

V7 package status: **10 tests passing**.
