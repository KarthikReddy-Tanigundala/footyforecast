# Methodology

## Evaluation protocol

- **Window:** 2017–2026 (2,063 games), the Squiggle era. Elo warms up from 1990; ML features start in 2012.
- **Walk-forward:** every round is tipped using only games *played* before its first game. Elo is updated in
  date order; the ML models are refit before every round; calibration maps are fitted on the previous three
  seasons' out-of-sample predictions.
- **Tuning discipline:** Elo is re-tuned each pre-season on the previous ten seasons. The classifier,
  hyperparameters and margin model were chosen on the 2014–2016 warm-up seasons only.
- **Selection rule:** the shipped model is the FootyForecast candidate with the most bits, fixed before
  evaluation. Picking the best of three on the scored window is a small optimistic bias, disclosed in the
  leaderboard; the 2027 live season is the clean test.
- **Metrics (Squiggle's):** tips (draws count as correct), **bits** (`1 + log2 p` right, `1 + log2(1−p)`
  wrong, `1 + ½·log2(p(1−p))` draw), margin MAE. Also Brier, log loss and ECE. Our bits and error reproduce
  Squiggle's published per-tip values for 100% / 99.9% of 45,249 tips.
- **Uncertainty:** round-clustered paired bootstrap (2,000 resamples) for model-vs-model bit differences,
  because games within a round are not independent.

## Elo

`E = 1 / (1 + 10^(−(R_h − R_a + HGA)/400))`. The update uses the home margin mapped linearly onto [0, 1] over
±80 points, `R' = R ± K(S − E)`, zero-sum. Ratings regress towards the league mean between seasons (carryover
≤ 0.9). 2020 margins are scaled ×1.25 for 16-minute quarters. Expansion clubs start at 1300.
`HGA = base + w_f·(log1p(home games at venue) − log1p(away games at venue)) + w_t·(log1p(away km) − log1p(home km))`,
using games in the last 730 days and great-circle distance from each club's base. An elo-diff → points
slope is fitted on the tuning seasons.

## Features and models

29 pre-match features (Elo gap and HGA, last 3/5 form, rest, ladder rank and percentage as they stood,
head-to-head, rolling stat dominance, selected-22 strength). All are attached with a strict "latest history row
before the round cutoff" join. The shipped classifier is a regularised logistic regression (C = 0.1); XGBoost is
reported as an ablation. The margin model is ridge regression. Draws are scored but not used as classifier labels.
Calibration is Platt scaling on logit p; isotonic is an ablation (it overfits ~600 calibration games).

## Explanations

The logistic model's SHAP values are exact: `φ_j = β_j · z_j` in log-odds (tested against
`shap.LinearExplainer`). Features are grouped into footy reasons; each reason is reported as the change in the
favourite's probability if that group's contribution were removed, through the same calibration map. The text
checks the raw data's direction against the model's: when they disagree (correlated features, e.g. recent margin
after Elo), the reason is worded as *discounted*, not as support for the tip.

## Archetypes

38 per-game stats per player-season (≥ 8 games at ≥ 50% time on ground), z-scored within season; PCA (90%
variance) → K-means; a Gaussian mixture gives soft memberships. K = 6 by silhouette within 6–9. Positions are the
validation (ARI 0.35, NMI 0.48), never an input. Names are matched from templates by the Hungarian algorithm.
UMAP is used only to draw the map.

## Known limitations

Annual-grain public data cannot support causal claims; named positions are noisy labels; the weekly job cannot
refresh player-level stats (see ARCHITECTURE.md); the market and the Squiggle Aggregate still outscore the
shipped model.
