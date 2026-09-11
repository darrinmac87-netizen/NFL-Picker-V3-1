from __future__ import annotations

from pathlib import Path
import pandas as pd

from .precision_layer import apply_precision_layer
from .pick_quality_gate import apply_pick_quality_gate
from .confidence_engine import apply_confidence_engine


def _find_column(df: pd.DataFrame, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def prepare_precision_backtest(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    ml_pick_col = _find_column(
        out,
        [
            "ml_pick",
            "v31_ml_pick",
            "final_v31_pick",
            "prediction",
            "predicted_winner",
        ],
    )

    prob_col = _find_column(
        out,
        [
            "ml_win_probability",
            "final_win_probability",
            "win_probability",
            "probability",
        ],
    )

    actual_col = _find_column(
        out,
        [
            "actual_winner",
            "winner",
            "result",
        ],
    )

    original_v3_col = _find_column(
        out,
        [
            "original_v3_pick",
            "v3_pick",
            "original_pick",
        ],
    )

    if ml_pick_col is None:
        raise ValueError("Historical ML pick column not found.")

    if prob_col is None:
        raise ValueError("Historical win probability column not found.")

    if actual_col is None:
        raise ValueError("Historical actual winner column not found.")

    out["final_v31_pick"] = out[ml_pick_col]

    out["final_win_probability"] = pd.to_numeric(
        out[prob_col],
        errors="coerce",
    )

    out["actual_winner"] = out[actual_col]

    if original_v3_col is not None:
        out["models_agree"] = (
            out[original_v3_col].astype(str)
            == out["final_v31_pick"].astype(str)
        )
        out["agreement_available"] = True
    else:
        out["models_agree"] = False
        out["agreement_available"] = False

    try:
        out = apply_confidence_engine(out)
    except Exception:
        pass

    try:
        out = apply_pick_quality_gate(out)
    except Exception:
        if "pick_quality_score" not in out.columns:
            prob = out["final_win_probability"].astype(float)

            if prob.max() <= 1.0:
                prob = prob * 100.0

            out["pick_quality_score"] = prob

    out = apply_precision_layer(out)

    out["correct"] = (
        out["final_v31_pick"].astype(str)
        == out["actual_winner"].astype(str)
    )

    return out


def precision_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for label in ["BEST PICK", "PLAY", "LEAN", "PASS"]:
        group = df[df["precision_recommendation"].eq(label)]

        games = len(group)
        correct = int(group["correct"].sum()) if games else 0
        wrong = games - correct

        accuracy = (
            round(correct / games * 100, 1)
            if games
            else None
        )

        rows.append(
            {
                "Precision_Level": label,
                "Games": games,
                "Correct": correct,
                "Wrong": wrong,
                "Accuracy": accuracy,
                "Avg_Precision_Score": (
                    round(group["precision_score"].mean(), 1)
                    if games
                    else None
                ),
            }
        )

    return pd.DataFrame(rows)


def weekly_top_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "season" not in df.columns or "week" not in df.columns:
        return pd.DataFrame()

    rows = []

    for n in [1, 2, 3, 5, 8, 10]:
        selected = (
            df.sort_values(
                ["season", "week", "precision_rank"]
            )
            .groupby(
                ["season", "week"],
                group_keys=False,
            )
            .head(n)
        )

        games = len(selected)
        correct = int(selected["correct"].sum()) if games else 0

        rows.append(
            {
                "Top_N_Per_Week": n,
                "Games": games,
                "Correct": correct,
                "Wrong": games - correct,
                "Accuracy": (
                    round(correct / games * 100, 1)
                    if games
                    else None
                ),
            }
        )

    return pd.DataFrame(rows)


def save_precision_backtest(
    historical_csv,
    output_dir,
    prefix="precision_backtest",
):
    historical_csv = Path(historical_csv)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = pd.read_csv(historical_csv)

    detail = prepare_precision_backtest(raw)
    summary = precision_summary(detail)
    weekly = weekly_top_summary(detail)

    readiness = pd.DataFrame(
        [
            {
                "Games": len(detail),
                "Has_Season": "season" in detail.columns,
                "Has_Week": "week" in detail.columns,
                "Agreement_Available": bool(
                    detail["agreement_available"].any()
                ),
                "BEST_PICK_Games": int(
                    detail["precision_recommendation"]
                    .eq("BEST PICK")
                    .sum()
                ),
                "PLAY_Games": int(
                    detail["precision_recommendation"]
                    .eq("PLAY")
                    .sum()
                ),
            }
        ]
    )

    detail.to_csv(
        output_dir / f"{prefix}_detail.csv",
        index=False,
    )

    summary.to_csv(
        output_dir / f"{prefix}_summary.csv",
        index=False,
    )

    weekly.to_csv(
        output_dir / f"{prefix}_weekly_top.csv",
        index=False,
    )

    readiness.to_csv(
        output_dir / f"{prefix}_readiness.csv",
        index=False,
    )

    return {
        "detail": detail,
        "summary": summary,
        "weekly": weekly,
        "readiness": readiness,
    }
