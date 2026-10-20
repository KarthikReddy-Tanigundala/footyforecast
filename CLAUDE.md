# CLAUDE.md — Project: FootyForecast
## AFL Match Prediction & Player Archetype Analytics
### Master Build Instructions for Claude Code (Opus 4.8)

---

## 0. PURPOSE & MISSION

You are building **FootyForecast** — a production-grade, fully-deployed, end-to-end AFL
(Australian Football League) analytics system that (1) predicts AFL match outcomes (calibrated
win probability + margin) and (2) clusters AFL players into playing-style archetypes, with model
explainability and an interactive deployed dashboard.

This is **supporting project 4 of 5** in Karthik Reddy Tanigundala's flagship data portfolio.
AFL is Melbourne's cultural heartbeat — this project is an instant talking point in any Melbourne
interview while demonstrating **classification, clustering, explainability, R, and statistical
modelling**. It targets Data Scientist / Data Analyst roles at Melbourne employers and is also
directly relevant to sports-analytics contexts (AFL clubs, Champion Data, betting firms like
Sportsbet/Tabcorp, sports-tech).

The project must read as the work of someone with **5–10 years of combined industry +
sports-analytics experience**: rigorous backtesting, calibrated probabilities, benchmarking
against public models, a clean reproducible pipeline, and a polished interactive dashboard.

**What separates this from the thousands of AFL tutorial clones (build ALL of these):**
1. **Benchmarking against Squiggle's public model leaderboard AND bookmaker odds** on Squiggle's
   own metrics — tips, **bits**, and **margin MAE**. This is the headline differentiator.
2. **An Elo rating system built from first principles** (not a library call) with venue/travel-
   aware home-ground advantage
3. **Calibrated win probabilities** (Platt/isotonic) with reliability diagrams and Brier score
4. **Rigorous walk-forward backtesting** respecting the AFL season structure, no leakage
5. **Player-archetype clustering** validated against ground-truth positions + a "find similar
   players" tool (engaging, shareable)
6. **SHAP explainability translated into plain-English footy reasoning** for non-technical fans
7. A live, public, weekly-auto-refreshed dashboard

**The output of this project is:**
1. A fully deployed, publicly accessible dashboard (Streamlit Community Cloud)
2. A polished GitHub repository with a standout README (prediction GIF + model-vs-Squiggle
   leaderboard + calibration plot + cluster visualisation)
3. A reproducible hybrid R-extract / Python-model pipeline
4. A GitHub Actions workflow that refreshes data and predictions weekly in-season
5. All assets demonstrating classification, clustering, explainability, and R proficiency

---

## 1. CRITICAL CONSTRAINTS & DESIGN DECISIONS (READ FIRST)

### 1.1 Free-tier only
The entire project runs free. Data (fitzRoy / Squiggle / AFL Tables / Footywire) is free and
public. Deployment (Streamlit Community Cloud) is free. GitHub Actions is free (unlimited minutes
on public repos). No paid API is required.

### 1.2 Architecture decision: HYBRID R-extract / Python-model (locked — ADR-001)
- **R touches data ONLY.** The `fitzRoy` R package is the single best AFL data source and has no
  equally-maintained Python equivalent. Use it to pull data once and cache to parquet/CSV.
- **Python does everything else** — features, Elo, models, calibration, clustering, SHAP,
  dashboard. Rationale: the Melbourne DS/DA market overwhelmingly asks for Python; SHAP and
  Streamlit are Python-native; Python apps deploy more easily on free tiers.
- **Do NOT use `reticulate`.** Keep the two languages decoupled via files (parquet). Cleaner and
  easier to deploy.
- An optional pure-R variant (tidymodels + Shiny) is documented for stats-role showcasing but is
  NOT the build target.

### 1.3 Data-source etiquette (non-negotiable)
| Source | Rule |
|---|---|
| **Squiggle API** (`api.squiggle.com.au`) | Set a meaningful User-Agent ("FootyForecast portfolio project <contact>"). The fetcher (not end users) pulls data. Pull whole seasons once and cache. NEVER poll aggressively or use as a live backend. It's a courtesy, not a production DB. |
| **fitzRoy sources** (AFL Tables, Footywire) | Scrapes live sites that occasionally change structure. Cache aggressively, pin fitzRoy version, handle failures gracefully. |
| **Champion Data** | Owns the richest AFL stats (official provider). NOT free, gated behind a licensed API. Do NOT assume access. Rely only on open Fryzigg/AFL Tables/Footywire data via fitzRoy. (The Fryzigg feed does expose a per-match `rating_points` field = AFL Player Ratings — the only ratings field available free.) |

### 1.4 Season-timing awareness
- AFL season runs ~March–September (~23 H&A rounds + finals, 18 teams).
- **Round 0 has existed since 2024** — handle round indexing carefully to avoid off-by-one
  leakage. Squiggle indexes pre-season as Round 0/−1.
- Out of season there are no new games — the dashboard must clearly show a "pre-season" state.

### 1.5 Benchmark honesty (a senior signal, not a weakness)
Public models and bookmakers are strong. "Tips" is a noisy metric — in some seasons the
wisdom-of-crowds Aggregate and even bookies beat every individual model on bits. Report bits/MAE
over MULTIPLE seasons; never cherry-pick a lucky season. If the ML model can't beat the Elo
benchmark, ship Elo as the headline and frame ML as an honest negative result.

---

## 2. HIGH-LEVEL ARCHITECTURE

```
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE A: DATA EXTRACTION (R, offline, cached)                        │
│                                                                        │
│  fitzRoy (R)                                                          │
│    ├── fetch_results / fetch_results_afltables  → match results       │
│    ├── fetch_fixture                            → upcoming fixtures   │
│    ├── fetch_ladder (source=squiggle)           → ladder             │
│    ├── fetch_lineup                             → team selections     │
│    ├── fetch_player_stats_fryzigg (81 cols)     → advanced stats      │
│    ├── fetch_squiggle_data("tips"/"sources")    → public model tips   │
│    └── fetch_betting_odds_footywire             → bookmaker odds       │
│         │                                                              │
│         ▼  write parquet (arrow)                                      │
│  data/raw/*.parquet  (immutable extracts)                            │
└──────────────────────────────────────────────────────────────────────┘
                          │  (decoupled via files — no reticulate)
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE B: PYTHON MODELLING & ANALYTICS                                │
│                                                                        │
│  MATCH PREDICTION                          PLAYER ARCHETYPES           │
│  ├── features.py (pre-match, no leakage)   ├── aggregate Fryzigg/game  │
│  ├── elo.py (from scratch, venue HGA)      ├── standardise → PCA/UMAP  │
│  ├── model_match.py (XGB win + margin)     ├── K-means / GMM           │
│  ├── calibration.py (Platt/isotonic)       ├── name + validate clusters│
│  └── benchmark.py (vs Squiggle+bookie)     └── similarity.py (compare) │
│                                                                        │
│  EXPLAINABILITY: explain.py (SHAP → plain-English footy reasons)      │
└──────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE C: BACKTESTING & EVALUATION                                   │
│  walk-forward by round/season (no leakage)                           │
│  metrics: tips, bits, margin MAE, log loss, Brier, calibration        │
│  → leaderboard vs baselines, bookmaker, Squiggle public models        │
└──────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PHASE D: DEPLOYMENT (Streamlit Community Cloud, free)               │
│  Tabs: This Round · Model vs The Field · Player Archetypes ·          │
│        Player Similarity                                              │
│  GitHub Actions cron: weekly in-season refresh (R extract → predict)  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. DATA EXTRACTION — R LAYER (`R/extract_data.R`)

### 3.1 Setup
```r
install.packages("fitzRoy")           # CRAN 1.7.0 (Depends R >= 4.1)
# or dev: remotes::install_github("jimmyday12/fitzRoy")
# Pin with renv: renv::init(); renv::snapshot()
library(fitzRoy)
library(arrow)   # parquet
library(dplyr)
```

### 3.2 Required pulls (write each to data/raw/ as parquet)

```r
# 1. Match results — training labels + Elo input
#    Full history (back to 1897) for Elo; modern window (2012+) for ML features
results <- fetch_results_afltables(1990:2026)
#    fields: home/away teams, scores, margin, venue, round, date

# 2. Fixtures — the upcoming round(s) to predict
fixture <- fetch_fixture(2026)

# 3. Ladder — at time of match (for ladder-position features)
ladder <- fetch_ladder(2026, source = "squiggle")

# 4. Lineups — player availability feature
lineup <- fetch_lineup(2026)

# 5. Advanced player stats — clustering + team-strength features (81 cols, player-match grain)
fryzigg <- fetch_player_stats_fryzigg(2012:2026)
#    includes rating_points (AFL Player Ratings), player_position (ground-truth labels)

# 6. Squiggle public model tips — the benchmark
sources <- fetch_squiggle_data("sources")
tips    <- fetch_squiggle_data("tips", year = 2015:2026)
#    SET User-Agent; pull whole seasons once and cache

# 7. Bookmaker odds — bookmaker benchmark
odds <- fetch_betting_odds_footywire()
```

### 3.3 Etiquette baked into the script
- Set `user_agent = "FootyForecast portfolio project <contact>"` on Squiggle calls.
- Pull whole seasons once; cache to parquet; never call inside a loop over users.
- Keep an offline historical store; during the season fetch only the latest round incrementally.
- Handle source failures gracefully (try/catch, log, exit non-zero so CI notices).

---

## 4. MATCH-OUTCOME MODELLING — DETAILED SPECS

### 4.1 Baselines (`benchmark.py`) — implement ALL, report tips/bits/MAE
```python
"""
- Home team always wins
- Higher-ladder team wins
- Bookmaker-implied probability (de-vig Footywire odds):
    p_home = (1/odds_home) / (1/odds_home + 1/odds_away)
- Squiggle 'Aggregate' tip as a reference ceiling
These define the bar every real model must clear.
"""
```

### 4.2 Elo rating system (`elo.py`) — the senior-credibility centrepiece
**Build from first principles so it can be explained in interview. Do NOT just call a library.**

```python
"""
- Each team starts at rating 1500.
- Expected home win probability:
    E_home = 1 / (1 + 10^(-(R_home - R_away + HGA)/400))
- HOME GROUND ADVANTAGE (start flat, then make venue/travel-aware):
    * Flat baseline: ~30-35 Elo points (a few goals)
    * Extend to: f(home venue is team's true home, travel distance, ground experience)
    * Real measurable AFL effects: Geelong @ Kardinia Park, Hawthorn @ UTAS/Launceston,
      Gold Coast in Darwin, both SA clubs @ Adelaide Oval are documented fortresses;
      shared MCG/Marvel tenants show smaller edges
- MARGIN-AWARE UPDATE:
    * map margin to [0,1] (normalise over [-80, 80])
    * R' = R + K * (S_actual - S_expected)
    * tune K (try 20-40; consider higher early-season K, lower later)
- SEASON CARRYOVER: regress ratings toward mean between seasons (carryover ~0.5-0.7)
  to handle list changes
- Convert Elo margin to predicted points margin via a fitted linear scaling
- Use Elo rating diff + HGA BOTH as features for the ML model AND as a standalone benchmark tipper
"""
```

### 4.3 ML win-probability classifier (`model_match.py`)
```python
"""
- Primary: XGBoost or LightGBM (objective='binary:logistic')
- Also fit logistic regression as an interpretable baseline
- Features (in features.py, all STRICTLY pre-match — no leakage):
    * Elo rating diff (home - away) + venue-adjusted HGA
    * Recent form: rolling win% and rolling avg margin over last 3/5 games per team
    * Ladder position & percentage at time of match
    * Home/away flag; venue; interstate-travel indicator + approx travel distance;
      ground experience (games at venue in last N)
    * Rest days since last match; short-break / 6-day-break flags
    * Head-to-head recent record
    * Team aggregate advanced-stat form from Fryzigg (rolling team means of contested
      possessions, clearances, inside 50s, metres gained, intercepts) — LAGGED to pre-match
    * Player-availability signal from lineups (count of high-rating_points players named)
- Tuning: cross-validate but RESPECT TIME ORDER (see backtesting)
"""
```

### 4.4 Margin regression (`model_match.py`)
```python
"""
- Separate XGBoost/LightGBM regressor (objective='reg:squarederror') predicting signed margin
- Report MAE to compare on Squiggle's MAE metric
- Optionally derive a second win-prob estimate from a distribution around predicted margin
  (Squiggle-style)
"""
```

### 4.5 Calibration (`calibration.py`)
```python
"""
- Wrap classifier in Platt scaling or isotonic regression
  (sklearn.calibration.CalibratedClassifierCV), fit on a held-out TIME block
- Report Brier score + reliability diagrams BEFORE and AFTER
- Calibrated probabilities are essential for the bits metric and are a senior differentiator
"""
```

---

## 5. BACKTESTING & EVALUATION — DETAILED SPECS (the rigour signal)

### 5.1 Backtesting (`backtest.py`)
```python
"""
Walk-forward / expanding-window by round and season:
  train on all data up to round R-1, predict round R, advance.
NEVER use future data. Respect AFL season structure (~23 H&A rounds + finals, 18 teams,
Round 0 since 2024). Handle round indexing carefully to avoid off-by-one leakage.
"""
```

### 5.2 Metrics (`metrics.py`) — compute per season AND overall
```python
"""
- TIPS: correct winners
- BITS: confidence-weighted metric from Monash's Probabilistic Footy Tipping competition,
  used on Squiggle's leaderboard. Rewards confident correct calls, punishes confident wrong ones.
    per game: 1 + log2(p)            if correct
              1 + log2(1-p)          if wrong
              1 + 0.5*log2(p*(1-p))  for draws
  (higher is better)
- MARGIN MAE: mean absolute difference between predicted and actual margins
- Log loss, Brier, accuracy, calibration error
"""
```

### 5.3 Benchmark table — FootyForecast vs the field
Compare against each baseline, the bookmaker, and named Squiggle public models. Reference points
from Squiggle's official 2024 leaderboard (for sanity-checking where a credible model lands):

| Metric | 2024 champion | Runner-up |
|---|---|---|
| Tips | @EloPredicts (156) | @thecruncherau (154) |
| Bits | @AflGlicko (35.76) | @MatterOfStats (33.24) |
| Margin MAE | @AflGlicko (24.91) | @MatterOfStats (25.09) |

A credible model lands roughly at bookmaker-level tip accuracy and competitive bits/MAE. **State
honestly where it sits.** Matching/beating the bookmaker on bits is the headline result if
achieved. (Historically, in some seasons no public model beat the bookies on tips or bits — so
framing matters.)

---

## 6. PLAYER CLUSTERING / ARCHETYPES — DETAILED SPECS

### 6.1 Data prep (`clustering.py`)
```python
"""
- Source: fetch_player_stats_fryzigg, aggregated to PER-GAME season averages per player
- Filter to players with >= a minimum games (e.g. 8) to reduce noise
- player_position (e.g. 'HBFL','WL','CHB','FPL') gives ground-truth labels for validation
"""
```

### 6.2 Feature set (~30–45 of the 81 columns — performance stats only)
```
kicks, handballs, disposals, effective_disposals, disposal_efficiency_percentage,
marks, contested_marks, marks_inside_fifty, marks_on_lead, intercept_marks,
tackles, tackles_inside_fifty, pressure_acts, def_half_pressure_acts,
clearances, centre_clearances, stoppage_clearances,
contested_possessions, uncontested_possessions, ground_ball_gets, f50_ground_ball_gets,
inside_fifties, rebounds, metres_gained,
goals, behinds, shots_at_goal, goal_assists, score_involvements, score_launches,
hitouts, hitouts_to_advantage, hitout_win_percentage, ruck_contests,
intercepts, spoils, one_percenters, clangers, turnovers, bounces,
player_height_cm, player_weight_kg, rating_points
```

### 6.3 Pipeline
```python
"""
- Standardise (StandardScaler)
- Optional dimensionality reduction: PCA for cluster stability, UMAP for the 2-D viz
- Cluster: K-means (primary) + GaussianMixture (soft-assignment alternative)
- Choose K via elbow (inertia) + silhouette; expect ~6-9 interpretable clusters
  (precedent: DFS Australia found 6; published AFL team-style work used K-means for 12 styles)
- time_on_ground_percentage available as a covariate; consider per-position z-scoring as a
  robustness variant (players rotate positions, so use per-game not per-minute)
"""
```

### 6.4 Interpretation & validation
```python
"""
- Profile each cluster's standardised feature means; NAME archetypes, e.g.:
  'Inside Midfielder', 'Outside/Winger', 'Rebounding Defender', 'Intercept/Key Defender',
  'Key Forward', 'Small/Pressure Forward', 'Ruck', 'Midfield-Forward'
- VALIDATE against player_position ground-truth labels
"""
```

### 6.5 Similarity tool (`similarity.py`)
```python
"""
- Cosine / Euclidean distance in standardised (or PCA) space
- 'Players most statistically similar to X'
- 'This young player resembles [star]'
- Highly engaging, shareable dashboard feature
"""
```

---

## 7. EXPLAINABILITY — DETAILED SPECS (`explain.py`)

```python
"""
- shap.TreeExplainer(model) on the XGBoost/LightGBM win model (exact, fast for trees)
  NOTE: XGBoost/LightGBM binary output is raw log-odds as a single array, while sklearn
  classifiers return per-class arrays — handle the shape difference.
- Produce:
    * global beeswarm/summary plot (expect Elo diff, form, HGA on top)
    * bar importance
    * dependence plots (e.g. Elo diff vs SHAP)
    * per-match waterfall/force plots
- TRANSLATE to plain English in the dashboard, e.g.:
  'Geelong favoured (72%): +18% from higher Elo rating, +6% home ground at Kardinia Park,
   -4% three-day rest disadvantage.'
- This explainable-prediction layer is a core wow factor for non-technical fans and signals
  communication skill to employers.
"""
```

---

## 8. DASHBOARD — DETAILED SPECS (`app/streamlit_app.py`)

```python
"""
Build in Streamlit. Tabs:

Tab 1 — This Round
  Upcoming fixtures with calibrated win probabilities, predicted margins, and the
  plain-English SHAP explanation per game.

Tab 2 — Model vs The Field
  Leaderboard table & charts comparing FootyForecast's tips/bits/MAE against bookmaker and
  Squiggle public models; calibration/reliability plot; season-by-season backtest curve.

Tab 3 — Player Archetypes
  Interactive UMAP/PCA scatter coloured by cluster, hover for player, filter by team/season;
  cluster profile panels.

Tab 4 — Player Similarity
  Pick a player → ranked most-similar players with stat radar comparison.

- Cache data/model loads with @st.cache_data / @st.cache_resource to stay within the
  guaranteed 1 GB RAM limit
- Keep model artifacts small (serialised boosters + cluster centroids), load lazily
- Show a clear 'pre-season' state when no games are scheduled
"""
```

---

## 9. WEEKLY REFRESH AUTOMATION (`.github/workflows/refresh.yml`)

```yaml
# Scheduled cron (e.g. "0 22 * * 1" Monday 22:00 UTC ~ Tuesday AM AEST, after each round)
# + workflow_dispatch. Public repo = unlimited free minutes.
# NOTE: GitHub schedules run in UTC and can be delayed under load.
#
# Steps:
# 1. Checkout
# 2. r-lib/actions/setup-r + restore renv
# 3. Run R/extract_data.R (incremental: latest round only)
# 4. Setup Python
# 5. Run feature build + regenerate predictions
# 6. Commit updated data/processed + models back to repo
#    (Streamlit auto-redeploys on push; Community Cloud rate-limits GitHub-triggered
#     updates to <=5/min)
# Secrets in GitHub Secrets; no keys needed but keep User-Agent set.
```

---

## 10. TECH STACK & PINNED VERSIONS

```
# R (data only)
R >= 4.1
fitzRoy 1.7.0
dplyr, arrow, tidyr, lubridate
# Lock with renv (renv.lock committed)

# Python >= 3.11 (everything else)
pandas
numpy
scikit-learn
xgboost
lightgbm
shap
umap-learn
matplotlib / plotly
streamlit
pyarrow
joblib
pytest
# Pin in requirements.txt; mind Streamlit's protobuf>=3.20,<6 constraint
```

**Optional pure-R variant (NOT the build target — documented for stats-role showcasing):**
tidymodels (`parsnip::boost_tree(engine="xgboost")`, `recipes`, `rsample`, `tune`, `yardstick`),
`elo` package, SHAP via `SHAPforxgboost`, deployed as Shiny on Posit Connect Cloud or
shinyapps.io (free tier: 5 apps, 25 active hours/month, 1 GB RAM).

---

## 11. REPOSITORY STRUCTURE

```
footyforecast/
├── CLAUDE.md
├── README.md                       ← THE most important file — see Section 13
├── ARCHITECTURE.md                 ← architecture + hybrid-design rationale
├── METHODOLOGY.md                  ← Elo, modelling, backtesting, metrics rationale
├── LICENSE                         ← MIT
├── renv.lock                       ← R deps lockfile
├── requirements.txt                ← pinned Python deps
├── .gitignore                      ← data/raw/ (if large), __pycache__/, .Rproj.user
│
├── R/
│   └── extract_data.R              ← fitzRoy pulls → data/raw/*.parquet
│
├── data/
│   ├── raw/                        ← immutable extracts (parquet)
│   └── processed/                  ← feature tables, latest predictions (committed)
│
├── src/footyforecast/
│   ├── __init__.py
│   ├── config.py                   ← settings (seasons, K-factor, HGA, thresholds)
│   ├── features.py                 ← match feature engineering (pre-match, no leakage)
│   ├── elo.py                      ← Elo rating system (from scratch, venue HGA)
│   ├── model_match.py              ← XGBoost win classifier + margin regressor
│   ├── calibration.py              ← Platt/isotonic + reliability diagrams
│   ├── benchmark.py                ← vs Squiggle + bookmaker, bits/MAE
│   ├── clustering.py               ← player archetypes
│   ├── similarity.py               ← player comparison tool
│   ├── explain.py                  ← SHAP → plain-English
│   └── evaluation/
│       ├── backtest.py             ← walk-forward by round/season
│       └── metrics.py              ← tips, bits, MAE, Brier, calibration
│
├── models/                         ← serialised artifacts (.joblib/.json + cluster centroids)
│
├── notebooks/                      ← EDA only (narrative, cleaned)
│   ├── 01_eda_matches.ipynb
│   ├── 02_elo_tuning.ipynb
│   └── 03_clustering.ipynb
│
├── app/
│   └── streamlit_app.py            ← 4-tab dashboard
│
├── results/
│   ├── leaderboard.md              ← FootyForecast vs baselines/bookie/Squiggle
│   ├── backtest_results.json
│   └── calibration_plot.png
│
├── assets/
│   ├── demo.gif                    ← this-round predictions walkthrough
│   ├── cluster_viz.png
│   └── screenshots/
│
├── tests/
│   ├── test_elo.py                 ← Elo update math correctness
│   ├── test_features.py            ← no leakage, correct lags
│   ├── test_metrics.py             ← bits/MAE/Brier correctness
│   └── test_backtest.py            ← walk-forward integrity
│
└── .github/workflows/
    ├── ci.yml                      ← lint + tests
    └── refresh.yml                 ← weekly in-season cron
```

---

## 12. BUILD ORDER (Claude Code Execution Sequence)

Follow this exact sequence. Validate each phase before proceeding.

```
PHASE 1 — DATA + BASELINES
  1.1  Create full repo structure (Section 11)
  1.2  Write .gitignore, LICENSE, requirements.txt (pinned), renv setup, config.py
  1.3  Write R/extract_data.R → cache all pulls to data/raw/ as parquet
       → Validate: parquet files populated, Squiggle User-Agent set, failures handled
  1.4  Write features.py (pre-match feature table, no leakage)
  1.5  Write evaluation/metrics.py (tips, bits, MAE, Brier) + test_metrics.py
  1.6  Write benchmark.py baselines (home-wins, higher-ladder, bookmaker de-vig, Squiggle Aggregate)
       → Validate: baseline benchmark table produced; metrics module tested
       → GATE: reproducible extract + baseline benchmark table complete

PHASE 2 — ELO + ML + CALIBRATION
  2.1  Write elo.py FROM SCRATCH (flat HGA first, then venue/travel-aware) + test_elo.py
       → Validate: Elo update math correct; ratings stable across seasons
  2.2  Write model_match.py (XGBoost win classifier + logistic baseline + margin regressor)
  2.3  Write calibration.py (Platt/isotonic, reliability diagrams, Brier before/after)
  2.4  Write evaluation/backtest.py (walk-forward by round/season) + test_backtest.py
       → Validate: full walk-forward backtest runs, no leakage
       → GATE: model bits >= bookmaker baseline; calibration error materially reduced
       → IF model can't beat Elo on bits: ship Elo as headline, frame ML as honest negative

PHASE 3 — CLUSTERING + EXPLAINABILITY
  3.1  Write clustering.py (aggregate Fryzigg → standardise → PCA/UMAP → K-means/GMM)
       → Validate: >=6 interpretable, position-validated clusters; named archetypes
  3.2  Write similarity.py (player comparison tool)
  3.3  Write explain.py (SHAP → plain-English footy reasons)
       → Validate: SHAP plots generate; explanations render in plain English
       → GATE: clusters validated against positions; SHAP explanations working

PHASE 4 — DASHBOARD + AUTOMATION + DOCS
  4.1  Train final models, serialise to models/, generate latest predictions to data/processed/
  4.2  Write app/streamlit_app.py (4 tabs)
       → Validate locally: streamlit run app/streamlit_app.py, all tabs load, <1GB RAM
  4.3  Write .github/workflows/refresh.yml (weekly cron) + ci.yml (lint + tests)
  4.4  Generate results/leaderboard.md + calibration_plot.png
  4.5  Write ARCHITECTURE.md, METHODOLOGY.md
  4.6  Clean notebooks (EDA, Elo tuning, clustering)
  4.7  Write README.md (full spec, Section 13)
       → GATE: live public URL auto-updating in-season

PHASE 5 — FINAL POLISH
  5.1  Record demo.gif (this-round predictions + explanations)
  5.2  Create cluster visualisation image + architecture diagram (Mermaid in README)
  5.3  Final verification: every link works, leaderboard current, all tests pass
  5.4  Deploy to Streamlit Cloud, confirm live URL, warm before demos
```

---

## 13. README.md — COMPLETE SPECIFICATION

```markdown
# 🏉 FootyForecast — Calibrated AFL Predictions & Player Archetype Analytics

> Calibrated AFL match predictions (win probability + margin) and player-archetype analytics,
> benchmarked against the public models and the bookmaker. Built with a hybrid R-extract /
> Python-model pipeline. Live demo below.

## 🎯 Live Demo
| Asset | Link |
|---|---|
| 🏉 Live Prediction Dashboard | [Open Dashboard](STREAMLIT_URL) |
| 🏆 Model vs The Field (leaderboard) | [View Results](results/leaderboard.md) |
| 📊 Calibration & Backtest | [View](results/calibration_plot.png) |

![FootyForecast Demo](assets/demo.gif)

## 📐 Architecture
[Mermaid diagram — R extract → parquet → Python modelling → backtest → Streamlit]
> Hybrid by design: R (fitzRoy) for data access only; Python for modelling, SHAP, and the
> dashboard. Decoupled via parquet (no reticulate).

## 🎯 What Makes This Different
- ✅ Benchmarked against Squiggle's public model leaderboard AND the bookmaker (tips, bits, MAE)
- ✅ Elo rating system built from first principles with venue/travel-aware home-ground advantage
- ✅ Calibrated win probabilities (reliability diagrams + Brier score)
- ✅ Rigorous walk-forward backtesting respecting AFL season structure (no leakage)
- ✅ Player-archetype clustering validated against ground-truth positions + similarity tool
- ✅ SHAP explanations translated into plain-English footy reasoning
- ✅ Weekly auto-refresh via GitHub Actions — the dashboard stays live in-season
- ✅ 100% free infrastructure (fitzRoy/Squiggle data, Streamlit Cloud)

## 🏆 Results
[leaderboard.md table + calibration plot]
> Honest note: public models and bookmakers are strong, and "tips" is a noisy metric. Results
> are reported over multiple seasons on bits and margin MAE, not a single lucky season.

## 🛠️ Tech Stack
[table: fitzRoy (R), XGBoost/LightGBM, scikit-learn, SHAP, UMAP, Streamlit, GitHub Actions]

## 📊 Data Sources & Ethics
[fitzRoy / Squiggle / AFL Tables / Footywire attribution + etiquette]
> Squiggle API used with a descriptive User-Agent and cached pulls — it's a courtesy service,
> not a production backend. Champion Data's licensed stats are NOT used.

## 🧠 Player Archetypes
[named clusters + UMAP visualisation + similarity examples]

## 🚀 Getting Started
[clone, renv restore, R extract, pip install, train, run dashboard locally]

## 🧪 Reproducing the Backtest
[python -m src.footyforecast.evaluation.backtest → regenerates leaderboard]

## 📚 Methodology
[link to METHODOLOGY.md]

## 🎓 About
Built by Karthik Reddy Tanigundala — RMIT Master of Data Science — targeting Data Scientist /
Data Analyst roles in Melbourne.
LinkedIn: linkedin.com/in/karthik-reddy-datascience
GitHub: github.com/KarthikReddy-Tanigundala
```

---

## 14. QUALITY STANDARDS — NON-NEGOTIABLE

### Code quality
- Type hints + docstrings on all Python functions; logging (not print)
- All settings (seasons, K-factor, HGA, thresholds) in config.py — not hardcoded
- R extract idempotent, failure-handled, pinned via renv
- Treat as production code, not a notebook dump

### Forecasting rigour
- Real measured numbers in the leaderboard — never fabricated
- Walk-forward backtesting only; respect AFL season structure; no leakage
- Calibrated probabilities with reliability diagrams + Brier score
- Honest benchmarking over multiple seasons; no cherry-picking

### Clustering rigour
- Clusters validated against ground-truth player_position labels
- K chosen by elbow + silhouette, not arbitrarily
- Archetypes named and profiled

### Data ethics (CRITICAL)
- Squiggle User-Agent set; cached pulls; never used as a live backend
- Champion Data licensed stats NOT used or assumed
- fitzRoy pinned; sources attributed in README

### Reproducibility
- renv.lock + requirements.txt committed; one-command setup documented
- Seeds set for clustering and any stochastic steps

---

## 15. INTERVIEW DEFENSIBILITY MAP

| Component | Likely question | Answer (verbatim) |
|---|---|---|
| Benchmarking | "How do you know your model is good?" | "I scored it on Squiggle's own metrics — tips, bits, and margin MAE — and compared it directly against Squiggle's public model leaderboard and the bookmaker's implied odds. Bits is a confidence-weighted metric from Monash's tipping competition: it rewards confident correct calls and punishes confident wrong ones. I report results over multiple seasons, not a lucky one." |
| Elo | "Walk me through your Elo system." | "Each team starts at 1500. Expected win probability is the logistic of the rating difference plus a home-ground advantage term. After each game I update ratings proportional to the surprise, scaled by a K-factor I tuned, and I made the update margin-aware by mapping margin onto a 0-to-1 scale. Between seasons I regress ratings toward the mean to handle list changes. The HGA isn't flat — it accounts for whether it's the team's true home venue, travel distance, and ground experience, because interstate travel is a real, measurable AFL effect." |
| Calibration | "Why calibrate probabilities?" | "A model can rank games well but still output overconfident probabilities. I wrapped the classifier in isotonic regression fit on a held-out time block and checked reliability diagrams and Brier score before and after. Calibrated probabilities matter because the bits metric — and any betting or risk use — depends on the probability being honest, not just the winner being right." |
| Leakage | "How did you avoid lookahead bias?" | "Strict walk-forward: I train on everything up to round R minus one and predict round R, then advance. All form and team-stat features are lagged to pre-match, ladder position is taken as it stood at match time, and I'm careful with AFL's Round 0 indexing since 2024 to avoid an off-by-one leak." |
| Clustering | "How did you validate the player clusters?" | "I clustered per-game advanced stats with K-means, chose K using the elbow and silhouette, then validated against the ground-truth position labels in the data — inside midfielders, key defenders, rucks and so on fell out cleanly. I also built a similarity tool using cosine distance so you can find the players most statistically like a given player." |
| Hybrid architecture | "Why R and Python?" | "fitzRoy is the best AFL data source and it's an R package with no equally-maintained Python equivalent, so R handles data extraction. Everything else — features, models, SHAP, the dashboard — is Python, because that's what the market asks for and what deploys cleanly for free. I keep them decoupled through parquet files rather than reticulate, which is simpler and more robust." |
| SHAP | "How do you explain a prediction?" | "I use SHAP's TreeExplainer on the gradient-boosted model and translate the top contributions into plain English — for example, 'Geelong favoured at 72%: plus 18% from a higher Elo rating, plus 6% home ground at Kardinia Park, minus 4% for a short rest.' That makes the model legible to a footy fan and shows I can communicate model behaviour to non-technical stakeholders." |

---

## 16. FINAL DELIVERABLES CHECKLIST

- [ ] GitHub repo: public, clean history
- [ ] README: prediction GIF + model-vs-Squiggle leaderboard + calibration plot + cluster viz + live link in first 25 lines
- [ ] Live Streamlit dashboard: 4 tabs working, within 1 GB RAM
- [ ] R extract: idempotent, parquet cache, Squiggle User-Agent set, failures handled
- [ ] Elo system: built from scratch, venue/travel HGA, unit-tested (test_elo.py)
- [ ] ML win classifier + margin regressor: calibrated, reliability diagrams + Brier reported
- [ ] Walk-forward backtest: leakage-safe, multi-season, reproducible
- [ ] Leaderboard: FootyForecast vs baselines/bookmaker/Squiggle with REAL numbers
- [ ] Player clusters: >=6 interpretable, position-validated, named archetypes
- [ ] Player similarity tool: working
- [ ] SHAP explanations: translated to plain English in dashboard
- [ ] GitHub Actions: CI tests + weekly in-season refresh
- [ ] Champion Data NOT used; Squiggle etiquette respected; sources attributed
- [ ] renv.lock + requirements.txt committed; one-command setup documented
- [ ] Streamlit app warmed before any interview
- [ ] Interview defensibility: Karthik can speak 60+ seconds on every component

---

*CLAUDE.md v1.0 — FootyForecast Project*
*Generated for Karthik Reddy Tanigundala — RMIT Master of Data Science 2026*
*Target: Melbourne Data Scientist / Data Analyst roles, AUD $90k–$160k band*
*Supporting Project 4 of 5 — the culturally resonant Melbourne piece (R + classification + clustering)*

---

## 17. ERRATA & BUILD DECISIONS (v1.1, 2026-10-20) — override §1–16 where they conflict

These came out of planning and the first build. Each is also an interview talking point.

| # | Spec said | Reality / decision | Why |
|---|---|---|---|
| 1 | fitzRoy 1.7.0, `arrow` for parquet | **fitzRoy 1.8.0** (CRAN, 2026-08-23) pinned in `renv.lock`; parquet via **nanoparquet** (already a fitzRoy import) | Current release; avoids the heavy `arrow` install on Windows |
| 2 | Squiggle UA "FootyForecast portfolio project <contact>" | UA = `FootyForecast/<ver> (+repo URL; <email>)`, email read from `FOOTY_CONTACT_EMAIL` in gitignored `.env` / GitHub Secret | Squiggle asks for a contact email; never hardcode it in a public repo |
| 3 | Walk-forward "by round" | Rounds are batches, but **every before/after check uses dates** (`cutoff_date` = round's first game; only results dated earlier are applied) | AFL Tables labels the 2024+ Opening Round "R1" while Squiggle calls it round 0 — joining or ordering on round numbers is a real off-by-one trap. Sources are joined on (date, unordered team pair) |
| 4 | Benchmark seasons 2015+ | **Evaluation window 2017–2026** | Squiggle tips start in 2017 (confirmed in the extract) |
| 5 | Bookmaker odds from Footywire | Footywire odds **stop in 2020**; Squiggle's **"Punters"** source is the market benchmark for 2017–2026; Footywire de-vig kept for 2010–2020 | Data availability |
| 6 | Tune Elo once | Elo is **re-tuned before every season on the trailing 10 seasons** (`results/elo_params_by_season.csv`); frozen-2000–16 and flat-HGA versions kept as ablations | Backtest the *process* the live model runs each pre-season, not a hindsight-picked parameter set |
| 7 | Elo carryover 0.5–0.7 | Carryover capped at **0.9** | Log-loss surface flat between 0.9 and 1.0 (Δ 0.0004); keep a mild regression for list turnover |
| 8 | Venue/travel HGA | HGA = base + w_fam·Δlog1p(games at venue, last 730 days) + w_travel·Δlog1p(km travelled) | Handles 2020 hubs / neutral venues / shared MCG tenants without special cases |
| 9 | 2020 | Margins ×1.25 for the Elo update (16-minute quarters); MAE reported unscaled | Comparable rating moves |
| 10 | — | 2026 introduced **Wildcard Finals** (AFL Tables round `WF`) | Handled naturally by round batching |
| 11 | Test fixtures | Match-level raw extracts (~1.2 MB) are **committed**; CI rebuilds everything offline and fails if `results/leaderboard.md` changes | Reproducibility from a fresh clone, no network/R/secrets in CI |
| 12 | Python ≥3.11 | Local 3.13; CI + Streamlit Cloud on **3.12**; deployed app uses lean `app/requirements.txt` | Free-tier RAM and install time |
| 13 | Season timing | Built in the 2026 off-season; app shows an **off-season state + replay of any 2017–2026 round**; feed `predictions/latest.json` designed for a **2027 Squiggle entry** | Live public track record from the 2027 Opening Round |

**Phase-1 result (honest):** walk-forward Elo 2017–2026 = 68.6% tips, 27.5 bits/season, MAE 26.79.
Behind all five external sources that tipped every game: Punters (34.2 bits/season), Matter of
Stats, Aggregate, Squiggle and Graft. Ahead of both ablations and both naive baselines. 2nd of 18
on Squiggle's 2019 bits table, but in the bottom quartile 2022–2026. Elo is under-confident at the
extremes (reliability diagram). Phase 2 (team-selection/player-rating features + calibration) targets this.

### 17.1 Phase 2 decisions (2026-10-20)

| # | Decision | Why |
|---|---|---|
| 14 | **Fryzigg has no 2026 data** → AFL-source player stats (`fetch_player_stats_afl`) fill 2025–26; Fryzigg stays primary for 2012–2025 | On the 2025 overlap: 99.8% of rows match, rating points agree 99.6%, metres gained 95.8%; AFL ids mapped onto Fryzigg ids (89% of 2026 player-games continue a history; most of the rest are debutants) |
| 15 | Features are attached **as-of** the round cutoff (`merge_asof`, strict `<`); perturbation test scrambles all results/stats after a date | The test caught a real leak: the debutant-rating fill was a global mean incl. future debuts → now an expanding as-of mean |
| 16 | ML trains on games **played** before the cutoff (not "earlier rounds") | The fold guard caught postponed games leaking in |
| 17 | Classifier, its hyperparameters and the margin model are chosen on **2014–2016 warm-up OOS predictions** only | Logistic (C=0.1) beat XGBoost's best (300 stumps) on warm-up log loss 0.558 vs 0.571 → logistic is the ML candidate, XGBoost an ablation; ridge (α=100) beat the XGBoost regressor on warm-up MAE 29.3 vs 30.0 |
| 18 | Calibration = **walk-forward Platt** on the previous 3 seasons' OOS predictions; isotonic kept as ablation | ~600 calibration games; isotonic overfits (−24 bits vs Platt on XGBoost) |
| 19 | Shipped model = most bits among {Elo, Elo+Platt, ML+Platt} over 2017–26 (rule fixed before running) | Disclosed as a small optimistic bias; 2027 live season is the clean test |
| 20 | Significance via **round-clustered paired bootstrap** | Games in a round aren't independent |

**Phase-2 result (honest):** shipped = logistic + Platt: 294.3 bits (29.4/season), MAE 26.39 — 5th of the
full-coverage field (behind Punters, Matter of Stats, Aggregate, Squiggle; ahead of Graft). vs Phase-1 Elo
+19.6 bits [−5.0, +43.8]; vs Elo+Platt +5.6 [−16.9, +28.4] (indistinguishable — most of the gain is
calibration: Platt alone gives Elo +14 bits). Market still clearly ahead (−48 [−71, −25]).
2022–26 Squiggle ranks improved from bottom quartile to 10th–14th.

### 17.2 Phase 3 decisions (2026-10-20)

| # | Decision | Why |
|---|---|---|
| 21 | Archetypes clustered on **38 per-game style stats only**; height/weight excluded | The AFL source (needed for 2026) has no physical data; style is the better definition of an archetype anyway |
| 22 | Player-season = games with **≥50% time on ground**, ≥8 such games; stats **z-scored within season** | Sub/injury cameos distort roles; era drift (e.g. tackle counts) shouldn't separate players |
| 23 | **K = 6** by best silhouette within 6–9 (position labels never used to choose K) | Matches the DFS Australia precedent in §6.3; silhouette 0.280 vs 0.244 at K=7 |
| 24 | Names via domain templates + **Hungarian assignment** on centroid profiles; "Key Defender" renamed **"Key / Intercept Defender"** | Reproducible naming; that cluster is 53% general defenders who intercept and spoil |
| 25 | GMM soft memberships mapped to K-means clusters by **max overlap (Hungarian)** | EM moves components away from their K-means initialisation; a bug showed primary = secondary otherwise |
| 26 | Explanations use **exact linear SHAP** (φ = β·z) stored per game in the walk-forward; reasons are counterfactual Δ pts through the season's calibration map | Exact, cheap, consistent with what was scored (guard asserts ≤1e-9 gap) |
| 27 | Reason text checks **data direction vs model direction**; disagreement → "edge discounted" | Correlated features give negative weights (recent margin after Elo = mean reversion); never invent a reason |
| 28 | `footyforecast.style` holds shared constants | Keeps the deployed app free of scikit-learn/SciPy/UMAP/SHAP |

**Phase-3 result:** 7,049 player-seasons → 6 archetypes; ARI 0.35 / NMI 0.48 vs named positions; Ruck
90% pure / 96% recall, Inside Midfielder 70% / 83%, Key Forward recall 75%; 27% hybrids. SHAP top drivers
agree across logistic and XGBoost (Elo gap, inside-50 dominance, ladder %, selected-22 strength).

### 17.3 Phase 4 decisions (2026-10-20)

| # | Decision | Why |
|---|---|---|
| 29 | R report fits strength via +1/-1 team-season design (Bradley-Terry style) with `lm`, then `lme4` `(1 \| venue)` | Independent of the Python Elo; result: HGA ~7.0 pts [6.0, 7.9], travel 1.41 pts per log-km unit (~11 pts for 2,700 km), between-venue SD 3.0 pts |
| 30 | `refresh.yml`: Tue 22:00 + Thu 02:00 UTC, Mar-Oct only, incremental extract, commits data/results/predictions | Idempotent; Thursday run follows team selection |
| 31 | Weekly job cannot refresh raw player stats (gitignored); new games use each team's last line-up | Documented limitation; official selections are future work |
