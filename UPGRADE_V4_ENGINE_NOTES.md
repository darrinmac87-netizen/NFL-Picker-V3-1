# v4.0 engine development build

Changes: fixed unfinished precision backtest return; added play-by-play pass/run EPA and success splits, sack rates, QB hit rates, and recorded interception/fumble lost rates; implemented opponent matchup edges. These are *team-level proxies*, not verified individual offensive-line or defensive-line grading.

**Not a trained or validated model:** new weights are provisional. No historical backtest or real-data end-to-end execution was performed in this environment. Injury, market and coaching contributions remain disabled. Missing PBP columns produce missing metrics; missing matchup edges are neutralized by the legacy diff() behavior. Never interpret missing data as confirmed zero.

Caution: turnover_rate uses play-level interception and fumble_lost when available. Verify the provider's column semantics, schedule alignment and 2026 coverage before deployment. V3.1 ML remains unchanged; this upgrade modifies the original V3 scoring branch.

Next: run walk-forward evaluation against original V3.1, calibrate probabilities, inspect data completeness and retrain before making accuracy claims.
