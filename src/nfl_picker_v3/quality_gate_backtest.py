from __future__ import annotations

from pathlib import Path
import pandas as pd

from .pick_quality_gate import apply_pick_quality_gate


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
