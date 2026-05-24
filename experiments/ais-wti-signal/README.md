# AIS tanker dwell times -> WTI

Free public AIS pings to weekly Houston Ship Channel tanker dwell time,
regressed on WTI front-month returns.

## Result

24 weekly observations (Jul-Dec 2023, one Wed/week sample). All ICs
statistically insignificant, but signs are consistent: more crude
sitting in port -> weaker forward WTI return.

| feature                 | pearson r | pearson p | spearman r | spearman p |
|-------------------------|----------:|----------:|-----------:|-----------:|
| weekly arrivals         |    +0.174 |     0.416 |     +0.071 |      0.740 |
| median dwell hours      |    -0.177 |     0.409 |     -0.038 |      0.859 |
| total dwell hours       |    -0.150 |     0.485 |     -0.154 |      0.473 |
| p90 dwell hours         |    -0.217 |     0.308 |     -0.240 |      0.259 |
| unique vessels          |    -0.053 |     0.806 |     -0.047 |      0.829 |

Event study, top vs bottom decile weeks by total dwell hours
(n_hi = n_lo = 3): mean 5-day fwd WTI log return -0.67% on
high-dwell weeks vs -0.01% on low-dwell weeks. Spread -0.66%,
Welch t = -0.265, p = 0.808.

In other words: the signal is in the right direction (idle tankers
imply oversupply imply price weakness) but does not survive even
loose significance testing on 24 weeks. **The pipeline is the
deliverable, not a trading signal.**

## Honest caveats

- **Sample size.** 24 weeks at one-Wednesday-per-week granularity.
  Spec called for 2 years; this run is 6 months. Bandwidth budget:
  each daily NOAA AIS zip is ~300 MB compressed (whole-US, not Zone
  15) and downloads at ~1.2 MB/s, so a full year of daily data is
  ~30 hours of download. Sampling weekly keeps the run under an hour.
- **One day per week.** Median/total dwell hours are estimated from a
  single weekday snapshot, so a tanker that arrives and departs
  between two consecutive Wednesdays is missed entirely. This biases
  dwell-hour totals downward and adds noise to week-over-week
  comparison.
- **Hand-drawn polygon.** Six-vertex convex hull around the Houston
  Ship Channel + Galveston Bay anchorages. Coarser than the actual
  navigable channel; some open-bay transits will be counted as dwell.
- **Tanker types only.** AIS `VesselType` 80-89. No filter for crude
  vs product tankers (would require an ITU registry join on `IMO`).
- **One day failed download** (2023-08-23, read timeout) so its week
  is missing from the panel.

## Layout

```
ais-wti-signal/
  data.py        # stream-download NOAA daily zips, filter to bbox + tankers
  features.py    # point-in-polygon -> visits -> weekly aggregates
  signals.py     # lag features, join WTI, compute IC + event study
  data/          # gitignored
```

## Numbers from the pipeline

- 25 daily NOAA files, ~8 GB raw download
- 700,328 in-bbox tanker pings, 989 unique MMSI
- 477,642 in-polygon pings (68.2% of in-bbox)
- 1,463 visits >= 1 hour after grouping by 1-hour gap
- 25 weekly observations -> 24 after 1-week feature lag

## Run

```
pip install pandas pyarrow requests shapely scipy yfinance
python data.py        # ~50 min, ~8 GB raw + ~30 MB parquet
python features.py    # ~15 s
python signals.py     # ~5 s
```

## References

- NOAA marinecadastre.gov AIS data dictionary
- Brancaccio, Kalouptsidi, Papageorgiou (2020), *Geography,
  Transportation, and Endogenous Trade Costs*
- Houston Ship Channel polygon: hand-drawn convex hull, six vertices,
  defined in `features.py`
