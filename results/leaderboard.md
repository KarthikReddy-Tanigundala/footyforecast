# FootyForecast vs The Field

Walk-forward backtest, **2017–2026** (2,063 games, 10 seasons). Every FootyForecast tip was made before its round using only earlier results, and before each season the Elo hyperparameters were re-tuned on the previous 10 seasons only, the same process the live model runs each pre-season.

Metrics follow Squiggle: **tips** = correct winners (draws count as correct), **bits** = Monash probabilistic score (higher is better; a coin flip scores 0), **MAE** = mean absolute margin error in points.

## Models that tipped every game

| # | Model | Type | Tips | Accuracy | Bits | Bits / season | MAE | Brier |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Punters | Market | 1,431 | 69.4% | 342.2 | 34.2 | 25.56 | 0.1963 |
| 2 | Matter of Stats | Squiggle model | 1,408 | 68.3% | 324.2 | 32.4 | 26.25 | 0.1988 |
| 3 | Aggregate | Crowd | 1,420 | 68.8% | 321.9 | 32.2 | 26.12 | 0.1986 |
| 4 | Squiggle | Squiggle model | 1,408 | 68.3% | 310.6 | 31.1 | 26.51 | 0.2003 |
| 5 | Graft | Squiggle model | 1,395 | 67.6% | 287.6 | 28.8 | 26.70 | 0.2034 |
| 6 | **FootyForecast Elo** | FootyForecast | 1,415 | 68.6% | 274.6 | 27.5 | 26.79 | 0.2048 |
| 7 | Elo (frozen 2000–16 params) | FootyForecast ablation | 1,407 | 68.2% | 271.8 | 27.2 | 27.06 | 0.2055 |
| 8 | Elo (flat HGA) | FootyForecast ablation | 1,392 | 67.5% | 268.8 | 26.9 | 26.92 | 0.2059 |
| 9 | Higher ladder | Baseline | 1,314 | 63.7% | 100.7 | 10.1 | – | 0.2309 |
| 10 | Home team | Baseline | 1,187 | 57.5% | 29.4 | 2.9 | 30.74 | 0.2428 |

## Season by season: FootyForecast vs the market and the crowd

| Season | FF tips | FF bits | FF MAE | Punters bits | Punters MAE | Aggregate bits | Aggregate MAE | FF rank (bits) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2017 | 129 | 17.8 | 29.61 | 22.9 | – | 21.3 | 29.40 | 8 / 10 |
| 2018 | 144 | 29.5 | 27.54 | 39.8 | – | 37.3 | 26.63 | 13 / 15 |
| 2019 | 138 | 24.7 | 27.39 | 23.3 | 26.79 | 23.1 | 26.88 | 2 / 18 |
| 2020 | 117 | 17.3 | 23.91 | 27.1 | 22.77 | 20.8 | 23.10 | 10 / 16 |
| 2021 | 135 | 19.1 | 27.84 | 18.6 | 27.06 | 19.7 | 26.90 | 8 / 19 |
| 2022 | 148 | 32.6 | 25.29 | 42.5 | 24.15 | 39.4 | 24.52 | 22 / 27 |
| 2023 | 150 | 27.0 | 25.41 | 34.0 | 25.13 | 31.0 | 25.34 | 18 / 27 |
| 2024 | 141 | 25.5 | 27.34 | 28.9 | 26.82 | 30.2 | 26.64 | 20 / 24 |
| 2025 | 159 | 39.3 | 27.63 | 51.7 | 26.16 | 47.7 | 26.24 | 24 / 24 |
| 2026 | 154 | 41.9 | 25.41 | 53.6 | 24.96 | 51.4 | 24.95 | 23 / 25 |

## Same-games comparison with every Squiggle model

Many public models only tipped some seasons, so each one is compared with FootyForecast on exactly the games both tipped. Positive Δ bits means FootyForecast scored more.

| Model | Seasons | Games | Their bits | FF bits | Δ bits | Their MAE | FF MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| Punters | 2017–2026 | 2,063 | 342.2 | 274.6 | -67.6 | 25.56 | 26.79 |
| s10 | 2018–2026 | 1,856 | 307.1 | 256.8 | -50.3 | 25.71 | 26.48 |
| Matter of Stats | 2017–2026 | 2,063 | 324.2 | 274.6 | -49.6 | 26.25 | 26.79 |
| Aggregate | 2017–2026 | 2,063 | 321.9 | 274.6 | -47.3 | 26.12 | 26.79 |
| Wheelo Ratings | 2022–2026 | 1,073 | 210.6 | 166.2 | -44.4 | 25.26 | 26.22 |
| Live Ladders | 2018–2026 | 1,856 | 298.0 | 256.8 | -41.2 | 25.97 | 26.48 |
| AFLalytics | 2019–2026 | 1,649 | 268.0 | 227.3 | -40.6 | 25.65 | 26.35 |
| Squiggle | 2017–2026 | 2,063 | 310.6 | 274.6 | -36.0 | 26.51 | 26.79 |
| Bookmaker (Footywire) | 2017–2020 | 756 | 120.0 | 85.5 | -34.4 | 26.10 | 27.44 |
| Drop Kick Data | 2022–2026 | 1,073 | 196.7 | 166.2 | -30.5 | 25.83 | 26.22 |
| Glicko Ratings | 2021–2026 | 1,280 | 208.0 | 185.3 | -22.7 | 26.06 | 26.49 |
| The Cruncher | 2021–2026 | 1,273 | 206.1 | 184.6 | -21.5 | 26.36 | 26.50 |
| Winnable | 2024–2026 | 650 | 127.4 | 106.6 | -20.8 | 26.22 | 26.79 |
| Massey Ratings | 2018–2026 | 1,821 | 271.9 | 251.7 | -20.3 | 26.47 | 26.56 |
| Cheap Stats | 2021–2026 | 1,277 | 203.3 | 184.7 | -18.6 | 26.40 | 26.50 |
| Don't Blame the Data | 2023–2026 | 866 | 151.8 | 133.6 | -18.2 | 26.37 | 26.45 |
| Data by Josh | 2022–2024 | 626 | 103.6 | 85.9 | -17.7 | 25.40 | 25.84 |
| Informed Stats | 2024–2026 | 650 | 123.0 | 106.6 | -16.5 | 27.03 | 26.79 |
| ZaphBot | 2021–2026 | 1,280 | 200.4 | 185.3 | -15.1 | 26.53 | 26.49 |
| The Wooden Finger | 2022–2026 | 1,073 | 179.3 | 166.2 | -13.1 | 26.62 | 26.22 |
| Graft | 2017–2026 | 2,063 | 287.6 | 274.6 | -13.0 | 26.70 | 26.79 |
| What Snoo Thinks | 2024–2026 | 628 | 121.4 | 108.5 | -12.8 | 26.61 | 26.79 |
| Elo Predicts! | 2022–2026 | 1,073 | 178.0 | 166.2 | -11.8 | 26.42 | 26.22 |
| AFL Lab | 2019–2026 | 1,649 | 237.5 | 227.3 | -10.2 | 26.09 | 26.35 |
| Hyperion | 2022–2026 | 1,059 | 173.4 | 164.6 | -8.8 | 26.80 | 26.25 |
| footycharts | 2024–2026 | 646 | 114.7 | 106.6 | -8.2 | 26.93 | 26.73 |
| In The Game | 2026–2026 | 218 | 49.7 | 41.9 | -7.8 | 26.31 | 25.41 |
| HBin | 2026–2026 | 218 | 48.6 | 41.9 | -6.8 | 25.24 | 25.41 |
| Swinburne | 2018–2020 | 576 | 78.1 | 71.5 | -6.6 | 26.69 | 26.46 |
| Figuring Footy | 2017–2017 | 207 | 19.7 | 17.8 | -1.8 | – | 29.61 |
| The Footycast | 2022–2026 | 1,071 | 167.3 | 165.8 | -1.5 | 26.14 | 26.21 |
| The Arc | 2017–2020 | 683 | 75.1 | 73.7 | -1.4 | 27.82 | 27.98 |
| AFL_GO | 2020–2020 | 162 | 18.4 | 17.3 | -1.1 | 23.31 | 23.91 |
| Elo (frozen 2000–16 params) | 2017–2026 | 2,063 | 271.8 | 274.6 | +2.9 | 27.06 | 26.79 |
| FMI | 2017–2024 | 1,628 | 188.7 | 193.7 | +5.0 | 27.36 | 26.88 |
| Elo (flat HGA) | 2017–2026 | 2,063 | 268.8 | 274.6 | +5.8 | 26.92 | 26.79 |
| Holy Grail Ratings | 2026–2026 | 218 | 35.3 | 41.9 | +6.6 | 25.31 | 25.41 |
| AFL Scorigami | 2022–2026 | 1,069 | 159.3 | 165.9 | +6.6 | 26.96 | 26.22 |
| Stattraction | 2018–2026 | 1,856 | 249.3 | 256.8 | +7.5 | 26.85 | 26.48 |
| AFL Gains | 2019–2019 | 207 | 12.4 | 24.7 | +12.3 | 27.79 | 27.39 |
| Fat Stats | 2019–2019 | 207 | 11.4 | 24.7 | +13.2 | 30.02 | 27.39 |
| The Flag | 2020–2023 | 792 | 77.6 | 96.1 | +18.4 | 26.39 | 25.71 |
| HPN | 2018–2019 | 414 | 29.3 | 54.2 | +24.9 | 29.46 | 27.46 |
| PlusSixOne | 2017–2026 | 2,050 | 237.1 | 274.4 | +37.2 | 27.07 | 26.75 |
| Higher ladder | 2017–2026 | 2,063 | 100.7 | 274.6 | +174.0 | – | 26.79 |
| Home team | 2017–2026 | 2,063 | 29.4 | 274.6 | +245.3 | 30.74 | 26.79 |

## Notes

- **Honest framing:** the betting market and the Squiggle Aggregate are strong. Tips are a noisy metric, so judge models on bits and MAE across all ten seasons, not one lucky year.
- Bookmaker (Footywire) odds are de-vigged proportionally and only exist for 2010–2020, so they appear in the same-games table; Squiggle's *Punters* source carries the market from 2017 to today.
- **Metric check:** our bits implementation matches Squiggle's published per-tip bits within 0.01 for 100.0% of 45,249 tips, and margin error within 0.5 points for 99.9%.
- 2020 was played with 16-minute quarters, so margins (and MAE) run smaller that season.
- Generated 2026-10-20 by `python -m footyforecast.evaluation.backtest`.
