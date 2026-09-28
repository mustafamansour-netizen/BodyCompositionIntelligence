# Body Composition Intelligence — V6.2

Streamlit app that imports a Withings export ZIP (or `weight.csv` + `other.csv`) and produces a practitioner-friendly, one-page body-composition report.

## What changed in V6.1 + V6.2

### History / trend display

- The history graph now uses the **full available horizontal plot width** instead of looking visually confined when switching between 8 and 12 points.
- 8 points show every date; denser histories automatically reduce date-label clutter while preserving every plotted data point.
- The latest history point is visually emphasized.
- The period-change block now shows the actual span in days rather than a generic “Period Δ”.
- The selected daily history rule still controls the one representative reading per measured day:
  - **Earliest complete scan**
  - **Latest complete scan**
  - **Daily median**
- The 12-month monthly medians remain based on **all complete scans**, remain gap-aware, and are never interpolated.

### Segmental analysis

- The torso value is now moved **inside the torso/chest area** in a high-contrast chip so it no longer sits over the head.
- Left/right arm and leg values are positioned closer to the silhouette.
- Each region now shows change versus the **previous fully complete segmental scan**.
- Muscle and fat arrows use coaching-oriented colour cues only:
  - muscle gain / fat loss = favourable direction,
  - muscle loss / fat gain = attention direction,
  - negligible change = neutral.
- Regional change is shown only when both the current and previous segmental snapshots are complete enough for a fair comparison.
- A compact regional summary highlights:
  - largest muscle gain,
  - largest fat loss,
  - the most relevant attention change (muscle loss or fat gain), when present.
- These changes are raw Withings BIA estimates and should be interpreted under consistent measurement conditions rather than as literal tissue change from one isolated scan.

### Practitioner-facing UI refinements

- Snapshot wording has been simplified to improve scan speed.
- Typography and small annotations were refined for better printed-PDF readability.
- Patient identity, KPI cards, practitioner snapshot, change summary, goal panel and one-page A4 structure from V6 are retained.
- Headline values always use the **latest complete whole-body scan**.
- “Previous measured day” deliberately ignores same-day repeats and respects the profile-selected daily-history rule.
- No proprietary InBody Score or SMM is invented. Withings Muscle Mass remains clearly identified as Withings total muscle mass.

## Why the report uses `pp`

`pp` means **percentage points**. It is intentionally retained because it is mathematically different from a relative percentage change.

Example: if body fat moves from **21.7% to 20.0%**, the change is:

- **−1.7 percentage points (`−1.7 pp`)**, not −1.7%.
- The relative percentage change would be approximately **−7.8%**: `(20.0 − 21.7) / 21.7 × 100`.

Using `pp` avoids mixing these two different concepts. In this report, `pp` is used when comparing percentage-based measurements such as body-fat percentage across scans.

## Segmental comparison rule

The current segmental view always shows the latest usable segmental snapshot. Change arrows are calculated only against the **previous fully complete segmental snapshot** containing all five regions for both muscle and fat:

- Left Arm
- Right Arm
- Torso
- Left Leg
- Right Leg

If no previous complete snapshot exists, the report shows no regional change rather than inventing one from partial data.

## PDF export on Streamlit Community Cloud

`requirements.txt` includes WeasyPrint. `packages.txt` lists the Debian system libraries commonly required by WeasyPrint on Streamlit Community Cloud. Commit both files to the repository root and reboot the app after dependency changes.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
pytest -q
```

V6.2 currently passes **8 tests**, covering core values, daily-history rules, patient identity, previous-day comparison, responsive history rendering, previous segmental snapshot selection, regional change rendering and practitioner-facing summaries.
