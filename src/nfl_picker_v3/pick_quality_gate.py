from __future__ import annotations

import pandas as pd


def quality_label(win_probability: float, confidence_tier: str | None = None, models_agree: bool = False) -> str:
    p = float(win_probability)
    tier = (confidence_tier or "").strip().upper()

    if models_agree and (tier == "ELITE" or p >= 0.75):
        return "BEST PICK"
    if models_agree and (tier == "STRONG" or p >= 0.67):
        return "PLAY"
    if tier in {"MEDIUM", "STRONG", "ELITE"} or p >= 0.60:
        return "LEAN"
    return "PASS"


def quality_score(win_probability: float, models_agree: bool = False, confidence_tier: str | None = None) -> int:
    p = float(win_probability)
    tier = (confidence_tier or "").strip().upper()

    score = p * 100.0
    if models_agree:
        score += 4.0

    tier_bonus = {
        "ELITE": 5.0,
        "STRONG": 3.0,
        "MEDIUM": 1.5,
        "LOW": 0.0,
        "PASS": -2.0,
    }
    score += tier_bonus.get(tier, 0.0)
    return int(round(max(0.0, min(100.0, score))))


def apply_pick_quality_gate(df: pd.DataFrame) -> pd.DataFrame:
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

    if "models_agree" in out.columns:
        agree = out["models_agree"].fillna(False).astype(bool)
    elif "agreement" in out.columns:
        agree = out["agreement"].astype(str).str.upper().eq("YES")
    else:
        agree = pd.Series(False, index=out.index)

    tiers = (
        out["confidence_tier"].fillna("").astype(str)
        if "confidence_tier" in out.columns
        else pd.Series("", index=out.index)
    )

    probs = pd.to_numeric(out[prob_col], errors="coerce").fillna(0.50)
    if probs.max() > 1.0:
        probs = probs / 100.0

    out["pick_quality_score"] = [
        quality_score(p, a, t)
        for p, a, t in zip(probs, agree, tiers)
    ]
    out["pick_quality"] = [
        quality_label(p, t, a)
        for p, a, t in zip(probs, agree, tiers)
    ]
    out["pick_priority_rank"] = (
        out["pick_quality_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    return out.sort_values(
        ["pick_priority_rank"],
        ascending=True,
    ).reset_index(drop=True)
