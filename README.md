# Body Composition Intelligence — v5

Streamlit app that imports a Withings export ZIP (or `weight.csv` + `other.csv`) and builds a printable one-page InBody-style body-composition report without inventing proprietary InBody metrics.

## v5 changes

- Replaced the assembled/cartoon body with a continuous male/female front-facing silhouette derived from the clean reference supplied for this iteration.
- The silhouette switches automatically with the profile sex.
- Segmental Muscle and Segmental Fat remain separate panels, with the five regional values around the body.
- Added **Daily history reading** to the profile:
  - Earliest complete scan
  - Latest complete scan
  - Daily median
- Added **History points shown**: 6 / 8 / 10 / 12 measured days.
- The daily-history setting affects only the recent InBody-style history graph.
- The 12-month monthly medians still use **all complete scans**, remain gap-aware, and are never interpolated.
- Headline metrics still always use the latest complete whole-body scan, independently of the selected history rule.
- Profile-driven goal weight remains fully editable and updates the projection automatically.

## Recommended history default

`Earliest complete scan` is the default because a consistent morning BIA measurement is often the most repeatable setup. If your personal routine is different, choose the rule that matches how you measure.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- Withings Muscle Mass is total muscle mass, not InBody Skeletal Muscle Mass (SMM).
- No proprietary InBody Score is calculated.
- Segmental values are Withings regional muscle/fat outputs; the report does not invent segment reference percentages.
