from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nfl_picker_v3.v31_optimizer import save_v31_outputs


def main():
    result = save_v31_outputs(ROOT / "outputs")

    print("V3.1 optimization complete")
    print("Selected C:", result["selected_C"])
    print("2025 holdout:", json.dumps(result["holdout_metrics"], indent=2))
    print("\nTop learned weights:")
    print(result["weights"].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
