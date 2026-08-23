from __future__ import annotations

from pathlib import Path
import pandas as pd

from .confidence_engine import confidence_tier, confidence_score


def _pick_probability_column(df: pd.DataFrame) -> str:
    """
    Find the best available V3.1 probability column in a historical result file.
    """
    candidates = [
        "ml_win_probability",
        "v31_win_probability",
        "final_probability",
        "win_probability",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "No supported probability column found. Expected one of: "
        "ml_win_probability, v31_win_probability, final_probability, win_probability."
    )


def _correct_column(df: pd.DataFrame) -> str:
    """
    Find the historical correctness column.
    """
    candidates = [
        "correct",
        "ml_correct",
        "final_correct",
        "v3_correct",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "No supported correctness column found. Expected one of: "
        "correct, ml_correct, final_correct, v3_correct."
    )


def _agreement_series(df: pd.DataFrame) -> pd.Series:
    """
    Determine whether Original V3 and ML/V3.1 agreed.
    """
    if "models_agree" in df.columns:
        return df["models_agree"].astype(bool)

    if "agreement" in df.columns:
        return df["agreement"].astype(str).str.upper().eq("YES")

    # Derive agreement when both pick columns exist.
    if "v3_pick" in df.columns and "ml_pick" in df.columns:
        return df["v3_pick"].astype(str) == df["ml_pick"].astype(str)

    if "original_v3_pick" in df.columns and "v31_pick" in df.columns:
        return (
            df["original_v3_pick"].astype(str)
            == df["v31_pick"].astype(str)
        )

    return pd.Series(False, index=df.index)


def prepare_confidence_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add confidence tier, confidence score, and agreement fields to
    historical prediction results.

    This function does not alter the historical winner or prediction.
    It only classifies already-existing predictions.
    """
    out = df.copy()

    prob_col = _pick_probability_column(out)
    correct_col = _correct_column(out)
    agree = _agreement_series(out)

    out["backtest_probability"] = pd.to_numeric(
        out[prob_col],
        errors="coerce",
    ).fillna(0.50)

    # Normalize 0-100 values to 0-1 if needed.
    if out["backtest_probability"].max() > 1.0:
        out["backtest_probability"] = (
            out["backtest_probability"] / 100.0
        )

    out["models_agree_backtest"] = agree

    out["confidence_score_backtest"] = [
        confidence_score(p, a)
        for p, a in zip(
            out["backtest_probability"],
            out["models_agree_backtest"],
        )
    ]

    out["confidence_tier_backtest"] = [
        confidence_tier(p, a)
        for p, a in zip(
            out["backtest_probability"],
            out["models_agree_backtest"],
        )
    ]

    out["backtest_correct"] = pd.to_numeric(
        out[correct_col],
        errors="coerce",
    ).fillna(0).astype(int)

    return out


def summarize_by_tier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return actual historical accuracy for PASS/LOW/MEDIUM/STRONG/ELITE.
    """
    d = prepare_confidence_backtest(df)

    order = ["PASS", "LOW", "MEDIUM", "STRONG", "ELITE"]

    summary = (
        d.groupby("confidence_tier_backtest", as_index=False)
        .agg(
            Games=("backtest_correct", "size"),
            Correct=("backtest_correct", "sum"),
            Accuracy=("backtest_correct", "mean"),
            Avg_Probability=("backtest_probability", "mean"),
        )
    )

    summary["Wrong"] = summary["Games"] - summary["Correct"]
    summary["Accuracy"] = (summary["Accuracy"] * 100).round(1)
    summary["Avg_Probability"] = (
        summary["Avg_Probability"] * 100
    ).round(1)

    summary = summary.rename(
        columns={"confidence_tier_backtest": "Tier"}
    )

    summary["Tier"] = pd.Categorical(
        summary["Tier"],
        categories=order,
        ordered=True,
    )

    return summary.sort_values("Tier").reset_index(drop=True)


def summarize_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare games where Original V3 and V3.1 agreed vs disagreed.
    """
    d = prepare_confidence_backtest(df)

    d["Agreement"] = d["models_agree_backtest"].map(
        {True: "AGREE", False: "DISAGREE"}
    )

    summary = (
        d.groupby("Agreement", as_index=False)
        .agg(
            Games=("backtest_correct", "size"),
            Correct=("backtest_correct", "sum"),
            Accuracy=("backtest_correct", "mean"),
            Avg_Probability=("backtest_probability", "mean"),
        )
    )

    summary["Wrong"] = summary["Games"] - summary["Correct"]
    summary["Accuracy"] = (summary["Accuracy"] * 100).round(1)
    summary["Avg_Probability"] = (
        summary["Avg_Probability"] * 100
    ).round(1)

    return summary[
        ["Agreement", "Games", "Correct", "Wrong", "Accuracy", "Avg_Probability"]
    ]


def summarize_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Measure actual accuracy at progressively stronger probability thresholds.
    """
    d = prepare_confidence_backtest(df)

    rows = []

    for threshold in [0.55, 0.60, 0.65, 0.70, 0.75]:
        subset = d[d["backtest_probability"] >= threshold]

        rows.append({
            "Minimum_Probability": f"{threshold * 100:.0f}%+",
            "Games": int(len(subset)),
            "Correct": int(subset["backtest_correct"].sum()) if len(subset) else 0,
            "Wrong": int(len(subset) - subset["backtest_correct"].sum()) if len(subset) else 0,
            "Accuracy": round(
                float(subset["backtest_correct"].mean()) * 100,
                1,
            ) if len(subset) else None,
        })

    return pd.DataFrame(rows)


def run_confidence_backtest(
    historical_csv: str | Path,
) -> dict[str, pd.DataFrame]:
    """
    Run the confidence analysis against an existing historical predictions CSV.

    The CSV must already contain predictions and actual outcomes.
    """
    historical_csv = Path(historical_csv)

    if not historical_csv.exists():
        raise FileNotFoundError(
            f"Historical predictions file not found: {historical_csv}"
        )

    df = pd.read_csv(historical_csv)

    return {
        "detail": prepare_confidence_backtest(df),
        "tiers": summarize_by_tier(df),
        "agreement": summarize_agreement(df),
        "thresholds": summarize_thresholds(df),
    }


def save_confidence_backtest(
    historical_csv: str | Path,
    output_dir: str | Path,
    prefix: str = "confidence_backtest",
) -> dict[str, pd.DataFrame]:
    """
    Run the analysis and save CSV reports.
    """
    result = run_confidence_backtest(historical_csv)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result["detail"].to_csv(
        output_dir / f"{prefix}_detail.csv",
        index=False,
    )
    result["tiers"].to_csv(
        output_dir / f"{prefix}_tiers.csv",
        index=False,
    )
    result["agreement"].to_csv(
        output_dir / f"{prefix}_agreement.csv",
        index=False,
    )
    result["thresholds"].to_csv(
        output_dir / f"{prefix}_thresholds.csv",
        index=False,
    )

    return result
