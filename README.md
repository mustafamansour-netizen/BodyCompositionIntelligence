# Body Composition Intelligence - rebuilt

A focused Streamlit app that imports a Withings data export and renders the approved one-page InBody-style body-composition report.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Input

Preferred: upload the original Withings export `.zip` directly.

The ZIP must contain:
- `weight.csv`
- `other.csv`

The app also supports uploading those two CSVs separately.

## Report logic

- Headline metrics: latest **complete** row in `weight.csv`.
- Segmental analysis: latest single snapshot containing segment muscle/fat values; no mixing of segment dates. Muscle and fat are shown as separate InBody-style panels with a smoother male/female body silhouette. Regional percentages are explicitly the share of the measured segment total, not an InBody reference score.
- Body Composition History: the latest **eight complete scans** are plotted in an InBody-style three-row history (Weight / Muscle Mass / Body Fat %) with the value printed at each point.
- The latest monthly median for each metric is shown beside the history row. The underlying 12-month monthly-median series stays gap-aware; missing months are never interpolated.
- Goal projection: assumes current fat-free mass is maintained. It is mathematical projection, not a target recommendation.
- Withings Muscle Mass is not relabeled as InBody Skeletal Muscle Mass (SMM), and no proprietary InBody Score is invented.
- ECW/TBW is calculated from exported ICW + ECW and marked as derived from rounded values.

## Why the app is deliberately small

The previous prototype split a simple report workflow across many pages. This rebuild keeps the user-facing workflow in one Streamlit page and puts parsing/calculation/rendering in one engine module. That makes it easier to validate the numbers and preserve the locked report design.

## If Withings changes export labels

Open **Data diagnostics** in the app. It shows all detected `other.csv` metric types and segment positions. The parser uses keyword matching for known Withings labels and will show `-` for an unmapped metric rather than fabricating a value.
