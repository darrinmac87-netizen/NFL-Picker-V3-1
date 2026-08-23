from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .engine import build_context
from .model import predict_game


FEATURES = [
    "edge_qb",
    "edge_offense",
    "edge_defense",
    "edge_trenches",
    "edge_injuries",
    "edge_market",
    "edge_turnovers",
    "edge_form",
    "edge_situational",
    "edge_matchup",
    "edge_coaching",
]


def _one_season(season: int) -> pd.DataFrame:
    # Current V3 snapshots only need previous season + current season-to-date.
    schedule, team_week, qb_week, injuries = build_context([season - 1, season])

    games = schedule[
        schedule["season"].eq(season)
        & schedule["home_score"].notna()
        & schedule["away_score"].notna()
    ].copy()

    rows = []
    for _, game in games.sort_values(["week", "gameday"]).iterrows():
        if game["home_score"] == game["away_score"]:
            continue

        pred = predict_game(game, schedule, team_week, qb_week, injuries)

        row = {
            "season": int(game["season"]),
            "week": int(game["week"]),
            "away_team": game["away_team"],
            "home_team": game["home_team"],
            "home_win": int(game["home_score"] > game["away_score"]),
        }
        for feature in FEATURES:
            value = pred.get(feature, 0.0)
            row[feature] = 0.0 if pd.isna(value) else float(value)

        rows.append(row)

    out = pd.DataFrame(rows)
    del schedule, team_week, qb_week, injuries, games
    gc.collect()
    return out


def build_dataset(start: int = 2020, end: int = 2025) -> pd.DataFrame:
    parts = []
    for season in range(start, end + 1):
        piece = _one_season(season)
        if not piece.empty:
            parts.append(piece)
        gc.collect()

    if not parts:
        return pd.DataFrame(
            columns=["season", "week", "away_team", "home_team", "home_win"] + FEATURES
        )
    return pd.concat(parts, ignore_index=True)


def _fit_model(train_df: pd.DataFrame, c_value: float) -> Pipeline:
    X = train_df[FEATURES].fillna(0.0)
    y = train_df["home_win"].astype(int)

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=float(c_value),
            penalty="l2",
            solver="liblinear",
            max_iter=2500,
            random_state=42,
        )),
    ])
    pipe.fit(X, y)
    return pipe


def _evaluate(model: Pipeline, df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    X = df[FEATURES].fillna(0.0)
    y = df["home_win"].astype(int)

    prob = model.predict_proba(X)[:, 1]
    detail = df.copy()
    detail["home_win_probability_ml"] = prob
    detail["ml_pick"] = np.where(prob >= 0.5, detail["home_team"], detail["away_team"])
    detail["actual_winner"] = np.where(detail["home_win"].eq(1), detail["home_team"], detail["away_team"])
    detail["correct"] = (detail["ml_pick"] == detail["actual_winner"]).astype(int)
    detail["ml_win_probability"] = np.where(prob >= 0.5, prob, 1.0 - prob)

    metrics = {
        "games": int(len(detail)),
        "correct": int(detail["correct"].sum()),
        "accuracy": float(accuracy_score(y, prob >= 0.5)),
        "brier_score": float(brier_score_loss(y, prob)),
        "log_loss": float(log_loss(y, prob)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y, prob))
    except Exception:
        metrics["roc_auc"] = None

    return metrics, detail


def optimize_and_build_production():
    """
    V3.1 experiment design:
      * 2020-2023: candidate training
      * 2024: validation used to choose regularization C
      * 2025: untouched final holdout
      * production: refit chosen model on 2020-2025 for 2026 live use
    """
    data = build_dataset(2020, 2025)

    base_train = data[data["season"].between(2020, 2023)].copy()
    validation = data[data["season"].eq(2024)].copy()
    holdout = data[data["season"].eq(2025)].copy()

    if base_train.empty or validation.empty or holdout.empty:
        raise ValueError("One or more required training/validation/holdout datasets are empty.")

    # Conservative grid to reduce overfitting.
    c_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

    trials = []
    best = None

    for c_value in c_grid:
        model = _fit_model(base_train, c_value)
        metrics, _ = _evaluate(model, validation)

        # Choose primarily by lower Brier score; use higher accuracy as tie-breaker.
        score = (metrics["brier_score"], -metrics["accuracy"])

        trials.append({
            "C": c_value,
            **metrics,
        })

        if best is None or score < best["score"]:
            best = {
                "score": score,
                "C": c_value,
                "validation_metrics": metrics,
            }

    selected_c = float(best["C"])

    # Final untouched 2025 holdout: train on 2020-2024, evaluate on 2025 once.
    pre_holdout_train = data[data["season"].between(2020, 2024)].copy()
    holdout_model = _fit_model(pre_holdout_train, selected_c)
    holdout_metrics, holdout_detail = _evaluate(holdout_model, holdout)

    # After evaluation, build production model for 2026 using all completed 2020-2025 data.
    production_train = data[data["season"].between(2020, 2025)].copy()
    production_model = _fit_model(production_train, selected_c)

    scaler = production_model.named_steps["scale"]
    clf = production_model.named_steps["clf"]

    coef = pd.Series(clf.coef_[0], index=FEATURES)
    importance = coef.abs()
    if importance.sum() > 0:
        importance = importance / importance.sum()

    weights = pd.DataFrame({
        "feature": FEATURES,
        "coefficient": coef.values,
        "importance": importance.values,
    }).sort_values("importance", ascending=False)

    weekly = (
        holdout_detail.groupby("week", as_index=False)
        .agg(
            Games=("correct", "size"),
            Correct=("correct", "sum"),
            Accuracy=("correct", "mean"),
        )
        .sort_values("week")
    )
    weekly["Wrong"] = weekly["Games"] - weekly["Correct"]
    weekly["Accuracy"] = (weekly["Accuracy"] * 100).round(1)

    production_payload = {
        "version": "V3.1",
        "features": FEATURES,
        "selected_C": selected_c,
        "training_seasons": [2020, 2021, 2022, 2023, 2024, 2025],
        "scaler_mean": [float(x) for x in scaler.mean_],
        "scaler_scale": [float(x) for x in scaler.scale_],
        "coefficients": [float(x) for x in clf.coef_[0]],
        "intercept": float(clf.intercept_[0]),
        "validation_2024": best["validation_metrics"],
        "holdout_2025": holdout_metrics,
    }

    return {
        "data": data,
        "trials": pd.DataFrame(trials),
        "selected_C": selected_c,
        "holdout_metrics": holdout_metrics,
        "holdout_detail": holdout_detail,
        "weekly": weekly,
        "weights": weights,
        "production_payload": production_payload,
    }


def save_v31_outputs(output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = optimize_and_build_production()

    result["trials"].to_csv(output_dir / "v31_validation_trials.csv", index=False)
    result["holdout_detail"].to_csv(output_dir / "v31_2025_holdout_games.csv", index=False)
    result["weekly"].to_csv(output_dir / "v31_2025_weekly_results.csv", index=False)
    result["weights"].to_csv(output_dir / "v31_learned_weights.csv", index=False)

    (output_dir / "v31_production_model.json").write_text(
        json.dumps(result["production_payload"], indent=2),
        encoding="utf-8",
    )

    summary = {
        "selected_C": result["selected_C"],
        "holdout_2025": result["holdout_metrics"],
    }
    (output_dir / "v31_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return result
