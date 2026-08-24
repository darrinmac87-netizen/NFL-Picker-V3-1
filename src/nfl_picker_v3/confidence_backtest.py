from __future__ import annotations

from pathlib import Path
import pandas as pd

from .confidence_engine import confidence_tier, confidence_score
from .model_agreement import apply_model_agreement


def _pick_probability_column(df):
    for col in ["ml_win_probability","v31_win_probability","final_probability","win_probability"]:
        if col in df.columns:
            return col
    raise ValueError("No supported probability column found.")


def _correct_column(df):
    for col in ["correct","ml_correct","final_correct","v3_correct"]:
        if col in df.columns:
            return col
    raise ValueError("No supported correctness column found.")


def _apply_historical_agreement(df):
    out = df.copy()

    if {"v3_pick","ml_pick"}.issubset(out.columns):
        return apply_model_agreement(out, original_col="v3_pick", ml_col="ml_pick")

    if {"original_v3_pick","v31_pick"}.issubset(out.columns):
        return apply_model_agreement(out, original_col="original_v3_pick", ml_col="v31_pick")

    out["agreement_count"] = 0
    out["agreeing_on"] = pd.NA
    out["agreement_level"] = "UNAVAILABLE"
    out["models_agree"] = pd.NA
    return out


def prepare_confidence_backtest(df):
    out = _apply_historical_agreement(df)
    prob_col = _pick_probability_column(out)
    correct_col = _correct_column(out)

    out["backtest_probability"] = pd.to_numeric(out[prob_col], errors="coerce").fillna(0.50)
    if out["backtest_probability"].max() > 1.0:
        out["backtest_probability"] = out["backtest_probability"] / 100.0

    agree = out["models_agree"].fillna(False).astype(bool)

    out["confidence_score_backtest"] = [
        confidence_score(p, a) for p, a in zip(out["backtest_probability"], agree)
    ]
    out["confidence_tier_backtest"] = [
        confidence_tier(p, a) for p, a in zip(out["backtest_probability"], agree)
    ]
    out["backtest_correct"] = pd.to_numeric(out[correct_col], errors="coerce").fillna(0).astype(int)
    return out


def summarize_by_tier(df):
    d = prepare_confidence_backtest(df)
    order = ["PASS","LOW","MEDIUM","STRONG","ELITE"]

    s = d.groupby("confidence_tier_backtest", as_index=False).agg(
        Games=("backtest_correct","size"),
        Correct=("backtest_correct","sum"),
        Accuracy=("backtest_correct","mean"),
        Avg_Probability=("backtest_probability","mean"),
    )
    s["Wrong"] = s["Games"] - s["Correct"]
    s["Accuracy"] = (s["Accuracy"] * 100).round(1)
    s["Avg_Probability"] = (s["Avg_Probability"] * 100).round(1)
    s = s.rename(columns={"confidence_tier_backtest":"Tier"})
    s["Tier"] = pd.Categorical(s["Tier"], categories=order, ordered=True)
    return s.sort_values("Tier").reset_index(drop=True)


def summarize_agreement(df):
    d = prepare_confidence_backtest(df)

    if d["agreement_level"].eq("UNAVAILABLE").all():
        return pd.DataFrame([{
            "Agreement":"UNAVAILABLE",
            "Games":int(len(d)),
            "Correct":int(d["backtest_correct"].sum()),
            "Wrong":int(len(d)-d["backtest_correct"].sum()),
            "Accuracy":round(float(d["backtest_correct"].mean())*100,1),
            "Avg_Probability":round(float(d["backtest_probability"].mean())*100,1),
        }])

    d["Agreement"] = d["models_agree"].map({True:"AGREE", False:"DISAGREE"})
    s = d.groupby("Agreement", as_index=False).agg(
        Games=("backtest_correct","size"),
        Correct=("backtest_correct","sum"),
        Accuracy=("backtest_correct","mean"),
        Avg_Probability=("backtest_probability","mean"),
    )
    s["Wrong"] = s["Games"] - s["Correct"]
    s["Accuracy"] = (s["Accuracy"]*100).round(1)
    s["Avg_Probability"] = (s["Avg_Probability"]*100).round(1)
    return s[["Agreement","Games","Correct","Wrong","Accuracy","Avg_Probability"]]


def summarize_thresholds(df):
    d = prepare_confidence_backtest(df)
    rows = []
    for t in [0.55,0.60,0.65,0.70,0.75]:
        x = d[d["backtest_probability"] >= t]
        rows.append({
            "Minimum_Probability":f"{t*100:.0f}%+",
            "Games":int(len(x)),
            "Correct":int(x["backtest_correct"].sum()) if len(x) else 0,
            "Wrong":int(len(x)-x["backtest_correct"].sum()) if len(x) else 0,
            "Accuracy":round(float(x["backtest_correct"].mean())*100,1) if len(x) else None,
        })
    return pd.DataFrame(rows)


def run_confidence_backtest(historical_csv):
    historical_csv = Path(historical_csv)
    if not historical_csv.exists():
        raise FileNotFoundError(f"Historical predictions file not found: {historical_csv}")
    df = pd.read_csv(historical_csv)
    return {
        "detail":prepare_confidence_backtest(df),
        "tiers":summarize_by_tier(df),
        "agreement":summarize_agreement(df),
        "thresholds":summarize_thresholds(df),
    }


def save_confidence_backtest(historical_csv, output_dir, prefix="confidence_backtest"):
    result = run_confidence_backtest(historical_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for key in ["detail","tiers","agreement","thresholds"]:
        result[key].to_csv(output_dir / f"{prefix}_{key}.csv", index=False)

    return result
