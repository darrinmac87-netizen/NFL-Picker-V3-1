# NFL Picker V3.1 CLEAN REINSTALL

Create ONE GitHub repository named `NFL-Picker-V3-1`.

Upload the CONTENTS of this folder directly to the repository root.

The root must show:
- app.py
- requirements.txt
- config.json
- train_v31_external.py
- src/
- data/
- outputs/
- .streamlit/
- .github/

Inside `src/nfl_picker_v3/` you must see:
- __init__.py
- data.py
- features.py
- model.py
- engine.py
- backtest_engine.py
- training.py
- v31_optimizer.py

Inside `.github/workflows/` you must see ONLY:
- train-v31.yml

There should be NO `train-v3.yml` file.

## Streamlit
Create ONE app:
- Repository: NFL-Picker-V3-1
- Branch: main
- Main file path: app.py

## First training run
GitHub -> Actions -> Train NFL Picker V3.1 -> Run workflow

Expected result files:
- outputs/v31_validation_trials.csv
- outputs/v31_2025_holdout_games.csv
- outputs/v31_2025_weekly_results.csv
- outputs/v31_learned_weights.csv
- outputs/v31_production_model.json
- outputs/v31_summary.json

Then refresh Streamlit.
