# ADR-002: Elo from first principles, venue/travel-aware

**Context.** An Elo is the standard strong baseline for AFL, and a library call hides the decisions that matter.

**Decision.** Implement Elo directly: margin-aware zero-sum updates, between-season regression, and home-ground
advantage built from ground familiarity and travel distance. Tune per pre-season on a rolling ten seasons.

**Consequences.** + Fully explainable; handles 2020 hubs and neutral venues without special cases; the R report
independently recovers the same travel effect (~10 points). Venue-aware HGA beat flat HGA by +3.0 bits and
re-tuning added +2.8. − More code to test (see `tests/test_elo.py`).
