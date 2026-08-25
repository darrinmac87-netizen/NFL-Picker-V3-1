from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nfl_picker_v3.quality_gate_backtest import save_quality_gate_backtest


def find_historical_file():
    candidates = [
        ROOT / "outputs" / "historical_2020_2024_games.csv",
        ROOT / "outputs" / "v31_2025_holdout_games.csv",
        ROOT / "outputs" / "ml_2025_holdout_games.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "No historical predictions CSV was found. Expected one of: "
        "outputs/historical_2020_2024_games.csv, "
        "outputs/v31_2025_holdout_games.csv, "
        "or outputs/ml_2025_holdout_games.csv."
    )


if __name__ == "__main__":
    historical_csv = find_historical_file()

    print(f"Using historical predictions: {historical_csv}")

    result = save_quality_gate_backtest(
        historical_csv=historical_csv,
        output_dir=ROOT / "outputs",
        prefix="quality_gate_backtest",
    )

    print("\nDATA READINESS")
    print(result["readiness"].to_string(index=False))

    print("\nQUALITY GATE RESULTS")
    print(result["quality"].to_string(index=False))

    print("\nBEST PICK ONLY")
    print(result["best_pick"].to_string(index=False))

    print("\nWEEKLY TOP RANKED PICKS")
    print(result["weekly_top_ranked"].to_string(index=False))

    print("\nQuality gate backtest reports saved in outputs/.")
