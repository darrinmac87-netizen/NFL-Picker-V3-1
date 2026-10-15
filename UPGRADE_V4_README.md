# DDD NFL Picker v4.0 — validation upgrade

This is a conservative upgrade to the uploaded V3.1 repository. It does NOT claim a new trained model or higher win rate.

## Changes
- Adds strict feature presence checks for the installed ML model and dimension validation.
- Adds weekly prediction and backtest CSV exports, usable in Excel.
- Allows 2026 week and season backtesting for completed games when the data source supports them.
- Repairs Pick History tab indentation so it appears regardless of model artifact availability.
- Adds model audit tab to disclose legacy placeholder features and train/test leakage risk.
- Preserves original production ML coefficients and feature semantics to avoid silently invalidating the model.

## Important limitations
- A 2025 holdout cannot be considered independent if the installed model was trained on 2025; inspect the training workflow and use walk-forward training.
- Current `trenches` is a team success-rate proxy, not line tracking. Current turnover edge is derived from QB edge.
- Do not add new inputs to the ML vector without retraining and independent validation.
- No claim of improved win percentage is made.

## Deploy
Upload changed `app.py` and this README to your GitHub repository, or replace the repository from this ZIP. Streamlit should redeploy automatically.
