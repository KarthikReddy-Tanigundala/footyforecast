"""Style-stat definitions shared by clustering, similarity and the dashboard.

Kept free of heavy dependencies (no scikit-learn/SciPy/UMAP) so the deployed app can import
it with only pandas and NumPy installed.
"""

# Per-game counting stats used for style (physical size is shown, never clustered on).
STYLE_STATS = [
    "kicks", "handballs", "disposals", "marks", "contested_marks", "marks_inside_fifty",
    "marks_on_lead", "intercept_marks", "tackles", "tackles_inside_fifty", "pressure_acts",
    "def_half_pressure_acts", "clearances", "centre_clearances", "stoppage_clearances",
    "contested_possessions", "uncontested_possessions", "ground_ball_gets", "f50_ground_ball_gets",
    "inside_fifties", "rebounds", "metres_gained", "goals", "behinds", "shots_at_goal",
    "goal_assists", "score_involvements", "score_launches", "hitouts", "hitouts_to_advantage",
    "ruck_contests", "intercepts", "spoils", "one_percenters", "clangers", "turnovers", "bounces",
    "rating_points",
]  # fmt: skip
FEATURES = [*STYLE_STATS, "disposal_efficiency"]

POSITION_GROUPS = {
    "FB": "Key Defender", "CHB": "Key Defender",
    "BPL": "General Defender", "BPR": "General Defender", "HBFL": "General Defender", "HBFR": "General Defender",
    "C": "Midfielder", "R": "Midfielder", "RR": "Midfielder",
    "WL": "Wing", "WR": "Wing",
    "RK": "Ruck",
    "FF": "Key Forward", "CHF": "Key Forward",
    "FPL": "General Forward", "FPR": "General Forward", "HFFL": "General Forward", "HFFR": "General Forward",
}  # fmt: skip
