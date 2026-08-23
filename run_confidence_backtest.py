from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nfl_picker_v3.confidence_backtest import save_confidence_backtest


def find_historical_file():
    candidates = [
        ROOT / "outputs" / "historical_2020_2024_games.csv",
        ROOT / "outputs" / "ml_2025_holdout_games.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "No historical predictions CSV was found. "
        "Expected outputs/historical_2020_2024_games.csv "
        "or outputs/ml_2025_holdout_games.csv."
    )


if __name__ == "__main__":
    historical_csv = find_historical_file()

    print(f"Using historical predictions: {historical_csv}")

    result = save_confidence_backtest(
        historical_csv=historical_csv,
        output_dir=ROOT / "outputs",
        prefix="confidence_backtest",
    )

    print("\nCONFIDENCE TIERS")
    print(result["tiers"].to_string(index=False))

    print("\nMODEL AGREEMENT")
    print(result["agreement"].to_string(index=False))

    print("\nPROBABILITY THRESHOLDS")
    print(result["thresholds"].to_string(index=False))

    print("\nConfidence backtest reports saved in outputs/.")
