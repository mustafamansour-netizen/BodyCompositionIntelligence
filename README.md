# Body Composition Intelligence — V6

Streamlit app that imports a Withings export ZIP (or `weight.csv` + `other.csv`) and produces a practitioner-friendly, one-page body-composition report.

## V6 highlights

- Stronger patient identity in the header, including optional **Client / Profile ID**.
- KPI cards now show movement versus the **previous measured day**.
- New **Practitioner Snapshot** row summarizing:
  - change since previous measured day,
  - position versus the 30-day median,
  - distance to the profile goal,
  - body-fat / muscle / BMI / visceral-fat reference status.
- Headline values always use the **latest complete whole-body scan**.
- “Previous measured day” uses the selected daily history rule (Earliest / Latest / Daily median), and deliberately ignores same-day repeats for the comparison.
- Segmental muscle and fat analysis retains separate male/female silhouettes and real Withings regional values.
- Body Composition History now uses the full report width and shows a period change for Weight, Muscle Mass and Body Fat %.
- New **Change Summary** panel: current versus previous measured day and current versus 30-day median.
- Reworked **Goal & Interpretation** panel with a Current → Goal visual, projected body fat at goal, current assessment and reference math.
- More compact scan-parameter panel and better use of A4 space while keeping the report on one page.
- No proprietary InBody Score or SMM is invented. Withings Muscle Mass remains clearly labelled as Withings total muscle mass.

## History rules

In the Streamlit profile, choose one value per measured day:

- **Earliest complete scan** (default)
- **Latest complete scan**
- **Daily median**

This setting affects the recent history display and the “previous measured day” comparison. The 12-month monthly medians are still calculated from **all complete scans**, remain gap-aware, and are never interpolated.

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

The V6 package currently passes 6 tests covering core values, daily-history rules, patient identity, previous-day comparison and practitioner-facing change summaries.
