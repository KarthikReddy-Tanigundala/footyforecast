# ADR-003: Walk-forward evaluation and leakage control

**Context.** Sports models leak easily: round numbers differ by source (the 2024+ Opening Round), games are
postponed, and rolling features can include the game being predicted.

**Decision.** Order everything by date. Each round's cutoff is its first game; models train on games *played*
before it; features attach via strict as-of joins. Guards assert it (fold integrity, perturbation tests that
scramble all results and stats after a date and require earlier features to be unchanged).

**Consequences.** + The tests caught two real leaks during the build (a debutant-rating fallback that used
future debuts, and postponed games entering training). − Slightly slower pipelines (refit per round, ~1 minute).
