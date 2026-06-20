# Pit-stop dashboard → moved

The interactive Streamlit dashboard for this model now lives in its own
repository, so it can be deployed on Streamlit Community Cloud and viewed
without cloning this monorepo:

**https://github.com/berezucc/f1-pit-stop-predictor**

That repo is the single source of truth for the dashboard: the app, the
trained `model.cbm` + `meta.json`, and the standalone `train_app_model.py`
that produces them from this competition's data.

## Why it's separate

The dashboard serves a single-row CatBoost model (OOF AUC 0.938) that scores
one lap at a time — the interactive cousin of the full stacked ensemble in
this directory (≈0.949), which needs whole-race sequence features and so
can't answer a single "pit on the next lap?" question.
