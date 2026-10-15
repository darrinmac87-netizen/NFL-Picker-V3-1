# V4.0 trench audit add-on

Adds an interactive team-level trench comparison panel to Streamlit. Uses only earlier weeks via `snap`, and explicitly reports missing statistics. Existing V3.1 ML picks remain unchanged; this is a diagnostic tool, **not evidence of improved accuracy**.

Caveats: `qb_hit` is a coarse pressure proxy; sack rate includes quarterback behavior, and run success rate is not a pure line metric. Provider column completeness must be checked. No proprietary player-level blocking grades are supplied. No verified 2026 walk-forward backtest has been performed.

Next: validate NFL play-by-play coverage, add licensed or documented line-specific win-rate metrics, and conduct paired walk-forward tests before modifying production weights.
