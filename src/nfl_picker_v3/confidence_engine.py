from __future__ import annotations

import pandas as pd


def confidence_tier(win_probability: float, models_agree: bool = False) -> str:
    p = float(win_probability)
    adjusted = p + (0.015 if bool(models_agree) else 0.0)

    if adjusted >= 0.75:
        return "ELITE"
    if adjusted >= 0.67:
        return "STRONG"
    if adjusted >= 0.60:
        return "MEDIUM"
    if adjusted >= 0.55:
        return "LOW"
    return "PASS"


def confidence_score(win_probability: float, models_agree: bool = False) -> int:
    p = float(win_probability)
    score = p * 100.0
    if bool(models_agree):
        score += 2.0
    return int(round(max(0.0, min(100.0, score))))


def apply_confidence_engine(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "final_probability" in out.columns:
        prob_col = "final_probability"
    elif "ml_win_probability" in out.columns:
        prob_col = "ml_win_probability"
    elif "v31_win_probability" in out.columns:
        prob_col = "v31_win_probability"
    elif "win_probability" in out.columns:
        prob_col = "win_probability"
    else:
        raise ValueError("No supported probability column found.")

    if "agreement" in out.columns:
        agree = out["agreement"].astype(str).str.upper().eq("YES")
    elif "models_agree" in out.columns:
        agree = out["models_agree"].astype(bool)
    else:
        agree = pd.Series(False, index=out.index)

    out["confidence_score"] = [
        confidence_score(p, a)
        for p, a in zip(out[prob_col].fillna(0.5), agree)
    ]
    out["confidence_tier"] = [
        confidence_tier(p, a)
        for p, a in zip(out[prob_col].fillna(0.5), agree)
    ]

    status_map = {
        "ELITE": "BEST PICK",
        "STRONG": "PLAY",
        "MEDIUM": "LEAN",
        "LOW": "CAUTION",
        "PASS": "PASS",
    }
    out["recommended_status"] = out["confidence_tier"].map(status_map)
    return out
