from __future__ import annotations

from pathlib import Path
import pandas as pd

from .pick_quality_gate import apply_pick_quality_gatefrom __future__ import annotations

from pathlib import Path
import pandas as pd

from .pick_quality_gate import apply_pick_quality_gate
from .model_agreement import apply_model_agreement
from .confidence_engine import apply_confidence_engine


def _find_correct_column(df: pd.DataFrame) -> str:
    for col in ["correct", "ml_correct", "final_correct", "v3_correct"]:
        if col in df.columns:
            return col
    raise ValueError(
        "No supported correctness column found. "
        "Expected one of: correct, ml_correct, final_correct, v3_correct."
    )


def _find_probability_column(df: pd.DataFrame) -> str:
    for col in [
        "final_probability",
        "ml_win_probability",
        "v31_win_probability",
        "win_probability",
    ]:
        if col in df.columns:
            return col
    raise ValueError("No supported probability column found.")


def _ensure_agreement(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """
    Reconstruct live-style agreement only when the historical file contains
    two independent pick columns.

    Returns:
        dataframe, agreement_available
    """
    out = df.copy()

    if {"v3_pick", "ml_pick"}.issubset(out.columns):
        out = apply_model_agreement(
            out,
            original_col="v3_pick",
            ml_col="ml_pick",
        )
        return out, True

    if {"original_v3_pick", "v31_pick"}.issubset(out.columns):
        out = apply_model_agreement(
            out,
            original_col="original_v3_pick",
            ml_col="v31_pick",
        )
        return out, True

    out["models_agree"] = False
    out["agreement_level"] = "UNAVAILABLE"
    return out, False


def _ensure_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add confidence fields if they are not already present.
    """
    out = df.copy()

    if "confidence_tier" not in out.columns:
        out = apply_confidence_engine(out)

    return out


def prepare_quality_gate_backtest(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """
    Prepare historical predictions for the same quality-gate logic used live.

    IMPORTANT:
    BEST PICK and PLAY depend on real model agreement. If the historical file
    lacks both independent pick columns, those categories cannot be validated
    honestly and agreement_available will be False.
    """
    out = df.copy()

    out, agreement_available = _ensure_agreement(out)
    out = _ensure_confidence(out)
    out = apply_pick_quality_gate(out)

    correct_col = _find_correct_column(out)

    out["backtest_correct"] = pd.to_numeric(
        out[correct_col],
        errors="coerce",
    ).fillna(0).astype(int)

    prob_col = _find_probability_column(out)
    out["backtest_probability"] = pd.to_numeric(
        out[prob_col],
        errors="coerce",
    ).fillna(0.50)

    if out["backtest_probability"].max() > 1.0:
        out["backtest_probability"] = out["backtest_probability"] / 100.0

    return out, agreement_available


def summarize_quality_gate(df: pd.DataFrame) -> pd.DataFrame:
    d, agreement_available = prepare_quality_gate_backtest(df)

    if not agreement_available:
        # Do not publish BEST PICK / PLAY performance claims when their required
        # agreement signal cannot be reconstructed.
        d = d[d["pick_quality"].isin(["PASS", "LEAN"])].copy()

    order = ["PASS", "LEAN", "PLAY", "BEST PICK"]

    summary = (
        d.groupby("pick_quality", as_index=False)
        .agg(
            Games=("backtest_correct", "size"),
            Correct=("backtest_correct", "sum"),
            Accuracy=("backtest_correct", "mean"),
            Avg_Quality_Score=("pick_quality_score", "mean"),
        )
    )

    summary["Wrong"] = summary["Games"] - summary["Correct"]
    summary["Accuracy"] = (summary["Accuracy"] * 100).round(1)
    summary["Avg_Quality_Score"] = summary["Avg_Quality_Score"].round(1)

    summary["pick_quality"] = pd.Categorical(
        summary["pick_quality"],
        categories=order,
        ordered=True,
    )

    summary = summary.sort_values("pick_quality").reset_index(drop=True)
    return summary.rename(columns={"pick_quality": "Quality"})


def summarize_weekly_top_ranked(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank games separately INSIDE each NFL week.

    This fixes the earlier mistake where Top 10 meant the ten strongest games
    across the entire 271-game season.
    """
    d, _ = prepare_quality_gate_backtest(df)

    if "week" not in d.columns:
        raise ValueError(
            "Historical file must contain a 'week' column for weekly ranking."
        )

    group_cols = ["week"]
    if "season" in d.columns:
        group_cols = ["season", "week"]

    # Re-rank inside each week.
    d["weekly_priority_rank"] = (
        d.groupby(group_cols)["pick_quality_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    rows = []

    for cutoff in [1, 2, 3, 5, 8, 10]:
        subset = d[d["weekly_priority_rank"] <= cutoff]

        rows.append({
            "Weekly_Top_Cutoff": f"Top {cutoff} per week",
            "Games": int(len(subset)),
            "Correct": int(subset["backtest_correct"].sum()) if len(subset) else 0,
            "Wrong": int(len(subset) - subset["backtest_correct"].sum()) if len(subset) else 0,
            "Accuracy": round(
                float(subset["backtest_correct"].mean()) * 100,
                1,
            ) if len(subset) else None,
        })

    return pd.DataFrame(rows)


def summarize_best_pick_only(df: pd.DataFrame) -> pd.DataFrame:
    d, agreement_available = prepare_quality_gate_backtest(df)

    if not agreement_available:
        return pd.DataFrame([{
            "Quality": "BEST PICK",
            "Games": 0,
            "Correct": 0,
            "Wrong": 0,
            "Accuracy": None,
            "Status": "UNAVAILABLE - historical file lacks independent V3 and ML pick columns",
        }])

    subset = d[d["pick_quality"].eq("BEST PICK")]

    if subset.empty:
        return pd.DataFrame([{
            "Quality": "BEST PICK",
            "Games": 0,
            "Correct": 0,
            "Wrong": 0,
            "Accuracy": None,
            "Status": "AVAILABLE - no games qualified",
        }])

    return pd.DataFrame([{
        "Quality": "BEST PICK",
        "Games": int(len(subset)),
        "Correct": int(subset["backtest_correct"].sum()),
        "Wrong": int(len(subset) - subset["backtest_correct"].sum()),
        "Accuracy": round(
            float(subset["backtest_correct"].mean()) * 100,
            1,
        ),
        "Status": "AVAILABLE",
    }])


def summarize_data_readiness(df: pd.DataFrame) -> pd.DataFrame:
    d, agreement_available = prepare_quality_gate_backtest(df)

    return pd.DataFrame([{
        "Games": int(len(d)),
        "Has_Season": "season" in d.columns,
        "Has_Week": "week" in d.columns,
        "Agreement_Available": agreement_available,
        "Can_Validate_BEST_PICK": agreement_available,
        "Can_Validate_PLAY": agreement_available,
    }])


def run_quality_gate_backtest(
    historical_csv: str | Path,
) -> dict[str, pd.DataFrame]:
    historical_csv = Path(historical_csv)

    if not historical_csv.exists():
        raise FileNotFoundError(
            f"Historical predictions file not found: {historical_csv}"
        )

    df = pd.read_csv(historical_csv)

    detail, _ = prepare_quality_gate_backtest(df)

    return {
        "detail": detail,
        "quality": summarize_quality_gate(df),
        "weekly_top_ranked": summarize_weekly_top_ranked(df),
        "best_pick": summarize_best_pick_only(df),
        "readiness": summarize_data_readiness(df),
    }


def save_quality_gate_backtest(
    historical_csv: str | Path,
    output_dir: str | Path,
    prefix: str = "quality_gate_backtest",
) -> dict[str, pd.DataFrame]:
    result = run_quality_gate_backtest(historical_csv)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result["detail"].to_csv(
        output_dir / f"{prefix}_detail.csv",
        index=False,
    )
    result["quality"].to_csv(
        output_dir / f"{prefix}_quality.csv",
        index=False,
    )
    result["weekly_top_ranked"].to_csv(
        output_dir / f"{prefix}_weekly_top_ranked.csv",
        index=False,
    )
    result["best_pick"].to_csv(
        output_dir / f"{prefix}_best_pick.csv",
        index=False,
    )
    result["readiness"].to_csv(
        output_dir / f"{prefix}_readiness.csv",
        index=False,
    )

    return result



def _find_correct_column(df: pd.DataFrame) -> str:
    for col in ["correct", "ml_correct", "final_correct", "v3_correct"]:
        if col in df.columns:
            return col
    raise ValueError(
        "No supported correctness column found. "
        "Expected one of: correct, ml_correct, final_correct, v3_correct."
    )


def prepare_quality_gate_backtest(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = apply_pick_quality_gate(out)

    correct_col = _find_correct_column(out)
    out["backtest_correct"] = pd.to_numeric(
        out[correct_col],
        errors="coerce",
    ).fillna(0).astype(int)

    return out


def summarize_quality_gate(df: pd.DataFrame) -> pd.DataFrame:
    d = prepare_quality_gate_backtest(df)

    order = ["PASS", "LEAN", "PLAY", "BEST PICK"]

    summary = (
        d.groupby("pick_quality", as_index=False)
        .agg(
            Games=("backtest_correct", "size"),
            Correct=("backtest_correct", "sum"),
            Accuracy=("backtest_correct", "mean"),
            Avg_Quality_Score=("pick_quality_score", "mean"),
        )
    )

    summary["Wrong"] = summary["Games"] - summary["Correct"]
    summary["Accuracy"] = (summary["Accuracy"] * 100).round(1)
    summary["Avg_Quality_Score"] = summary["Avg_Quality_Score"].round(1)

    summary["pick_quality"] = pd.Categorical(
        summary["pick_quality"],
        categories=order,
        ordered=True,
    )

    summary = summary.sort_values("pick_quality").reset_index(drop=True)

    return summary.rename(columns={"pick_quality": "Quality"})


def summarize_top_ranked(df: pd.DataFrame) -> pd.DataFrame:
    d = prepare_quality_gate_backtest(df)

    rows = []

    for cutoff in [1, 3, 5, 8, 10]:
        subset = d[d["pick_priority_rank"] <= cutoff]

        rows.append({
            "Top_Rank_Cutoff": f"Top {cutoff}",
            "Games": int(len(subset)),
            "Correct": int(subset["backtest_correct"].sum()) if len(subset) else 0,
            "Wrong": int(len(subset) - subset["backtest_correct"].sum()) if len(subset) else 0,
            "Accuracy": round(
                float(subset["backtest_correct"].mean()) * 100,
                1,
            ) if len(subset) else None,
        })

    return pd.DataFrame(rows)


def summarize_best_pick_only(df: pd.DataFrame) -> pd.DataFrame:
    d = prepare_quality_gate_backtest(df)
    subset = d[d["pick_quality"].eq("BEST PICK")]

    if subset.empty:
        return pd.DataFrame([{
            "Quality": "BEST PICK",
            "Games": 0,
            "Correct": 0,
            "Wrong": 0,
            "Accuracy": None,
        }])

    return pd.DataFrame([{
        "Quality": "BEST PICK",
        "Games": int(len(subset)),
        "Correct": int(subset["backtest_correct"].sum()),
        "Wrong": int(len(subset) - subset["backtest_correct"].sum()),
        "Accuracy": round(
            float(subset["backtest_correct"].mean()) * 100,
            1,
        ),
    }])


def run_quality_gate_backtest(
    historical_csv: str | Path,
) -> dict[str, pd.DataFrame]:
    historical_csv = Path(historical_csv)

    if not historical_csv.exists():
        raise FileNotFoundError(
            f"Historical predictions file not found: {historical_csv}"
        )

    df = pd.read_csv(historical_csv)

    return {
        "detail": prepare_quality_gate_backtest(df),
        "quality": summarize_quality_gate(df),
        "top_ranked": summarize_top_ranked(df),
        "best_pick": summarize_best_pick_only(df),
    }


def save_quality_gate_backtest(
    historical_csv: str | Path,
    output_dir: str | Path,
    prefix: str = "quality_gate_backtest",
) -> dict[str, pd.DataFrame]:
    result = run_quality_gate_backtest(historical_csv)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result["detail"].to_csv(
        output_dir / f"{prefix}_detail.csv",
        index=False,
    )
    result["quality"].to_csv(
        output_dir / f"{prefix}_quality.csv",
        index=False,
    )
    result["top_ranked"].to_csv(
        output_dir / f"{prefix}_top_ranked.csv",
        index=False,
    )
    result["best_pick"].to_csv(
        output_dir / f"{prefix}_best_pick.csv",
        index=False,
    )

    return result
