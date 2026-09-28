# Body Composition Intelligence — v4

Streamlit app that imports a Withings export ZIP (or weight/other CSVs) and builds a printable one-page InBody-style body-composition report without inventing proprietary InBody metrics.

## v4 changes

- Refined segmental muscle/fat panels with a smoother anatomical silhouette.
- Removed the white pelvis/feet treatment and floating torso card.
- Removed segment-share percentages that could be visually over-interpreted.
- Softer fat-distribution palette; colour is regional identification, not a clinical rating.
- More compact segmental section and a wider/larger history panel.
- InBody-style history uses the latest **real complete scan per calendar day**, so repeated same-day dates no longer clutter the chart.
- History value labels are no longer horizontally stretched.
- Scan-data summary added to Body Scan Parameters.
- Profile-driven goal weight remains unchanged and updates the goal maths automatically.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- Withings Muscle Mass is total muscle mass, not InBody Skeletal Muscle Mass (SMM).
- No proprietary InBody Score is calculated.
- Missing monthly trend months remain gaps; values are not interpolated.
