"""Pregame team-level trench matchup audit; not player grading."""
import pandas as pd
from .features import snap

METRICS = {
    "sack_allowed_rate": "Offensive pass protection (sack rate allowed)",
    "sack_generated_rate": "Defensive pass rush (sack rate generated)",
    "qb_hit_allowed_rate": "Offensive QB-hit rate allowed (proxy)",
    "qb_hit_generated_rate": "Defensive QB-hit rate generated (proxy)",
    "run_success": "Offensive rushing success rate (proxy)",
    "run_success_allowed": "Defensive rushing success allowed (proxy)",
}

def audit_matchup(tw, qw, season, week, home, away, recent=4):
    """All snapshots strictly precede requested week. Missing data remains missing."""
    h = snap(tw, qw, season, week, home, recent)
    a = snap(tw, qw, season, week, away, recent)
    rows = []
    for metric, label in METRICS.items():
        hv, av = h.get(metric, float('nan')), a.get(metric, float('nan'))
        rows.append({"Metric":label, f"{home}":hv, f"{away}":av,
                     "Available for both":bool(pd.notna(hv) and pd.notna(av))})
    return pd.DataFrame(rows)

def matchup_edges(tw, qw, season, week, home, away, recent=4):
    """Home advantage expressed as percentage-point differences; None means unavailable."""
    h = snap(tw, qw, season, week, home, recent)
    a = snap(tw, qw, season, week, away, recent)
    pairs = {
        "Home protection vs away pass rush": (a.get('sack_generated_rate'),h.get('sack_allowed_rate')),
        "Away protection vs home pass rush": (h.get('sack_generated_rate'),a.get('sack_allowed_rate')),
        "Home run offense vs away run defense": (h.get('run_success'),a.get('run_success_allowed')),
        "Away run offense vs home run defense": (a.get('run_success'),h.get('run_success_allowed')),
    }
    return {k: (round(100*(x-y),2) if pd.notna(x) and pd.notna(y) else None) for k,(x,y) in pairs.items()}
