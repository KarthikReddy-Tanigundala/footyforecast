# ADR-004: Squiggle feed and API etiquette

**Context.** The project targets a public 2027 Squiggle leaderboard entry; Squiggle is a free community service
that asks for an identifying User-Agent with a contact email and cached, polite requests.

**Decision.** Only the extract step calls Squiggle (whole seasons, ≥ 1 s apart, cached, email from
`FOOTY_CONTACT_EMAIL`). Tips are published as `predictions/latest.json` (Squiggle field names) with an immutable
per-round archive, so the track record is auditable. The dashboard never calls any API.

**Consequences.** + Verifiable public record; no live dependency. − The email lives in a gitignored `.env` /
GitHub Secret, so contributors must set it to run the extract.
