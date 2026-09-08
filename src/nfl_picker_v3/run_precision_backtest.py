from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nfl_picker_v3.precision_backtest import save_precision_backtest


def find_historical_file():
    candidates = [
        ROOT / "outputs" / "v31_2025_holdout_games.csv",
        ROOT / "outputs" / "ml_2025_holdout_games.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "No historical V3.1 holdout CSV found. Expected one of: "
        "outputs/v31_2025_holdout_games.csv or "
        "outputs/ml_2025_holdout_games.csv."
    )


if __name__ == "__main__":
    historical_csv = find_historical_file()

    print(f"Using historical predictions: {historical_csv}")

    result = save_precision_backtest(
        historical_csv=historical_csv,
        output_dir=ROOT / "outputs",
        prefix="precision_backtest",
    )

    print("\nDATA READINESS")
    print(result["readiness"].to_string(index=False))

    print("\nPRECISION LAYER RESULTS")
    print(result["summary"].to_string(index=False))

    print("\nWEEKLY TOP RANKED PICKS")
    print(result["weekly"].to_string(index=False))

    print("\nPrecision backtest reports saved in outputs/.")
