from pathlib import Path
import sys
import json
import math

import numpy as np
import pandas as pd
import streamlit as st
from pick_history_ui import render_pick_history

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nfl_picker_v3.engine import weekly_predictions
from nfl_picker_v3.backtest_engine import backtest

st.set_page_config(page_title="NFL Picker V3.1", page_icon="🏈", layout="wide")
st.title("🏈 NFL Picker V3.1")
st.caption("Original V3 + optimized production ML model for 2026")

tabs = st.tabs([
    "2026 Weekly Picks",
    "Week Backtest",
    "Season Backtest",
    "V3.1 Model Results",
    "Pick History"])

MODEL_PATH = ROOT / "outputs" / "v31_production_model.json"


def _sigmoid(x):
    x = max(min(float(x), 20.0), -20.0)
    return 1.0 / (1.0 + math.exp(-x))


def _ml_probability(row, payload):
    features = payload["features"]
    mean = np.array(payload["scaler_mean"], dtype=float)
    scale = np.array(payload["scaler_scale"], dtype=float)
    coef = np.array(payload["coefficients"], dtype=float)
    intercept = float(payload["intercept"])

    values = np.array([
        0.0 if pd.isna(row.get(f, 0.0)) else float(row.get(f, 0.0))
        for f in features
    ], dtype=float)

    safe_scale = np.where(scale == 0, 1.0, scale)
    z = (values - mean) / safe_scale
    logit = intercept + float(np.dot(coef, z))
    return _sigmoid(logit)


def _confidence(prob):
    p = max(prob, 1.0 - prob)
    if p >= 0.75: return 5
    if p >= 0.67: return 4
    if p >= 0.60: return 3
    if p >= 0.55: return 2
    return 1


with tabs[0]:
    st.subheader("2026 Weekly Picks")

    c1, c2 = st.columns(2)
    season = c1.number_input("Season", 2020, 2035, 2026, key="weekly_season")
    week = c2.number_input("Week", 1, 18, 1, key="weekly_week")

    if not MODEL_PATH.exists():
        st.warning(
            "V3.1 production model is not installed yet. Run the GitHub Action "
            "'Train NFL Picker V3.1' first."
        )

    if st.button("Generate V3.1 Picks"):
        try:
            df = weekly_predictions(int(season), int(week))

            # Keep original V3 result.
            df["v3_pick"] = df["pick"]
            df["v3_probability"] = df["win_probability"]

            if MODEL_PATH.exists():
                payload = json.loads(MODEL_PATH.read_text(encoding="utf-8"))

                home_probs = df.apply(lambda r: _ml_probability(r, payload), axis=1)
                df["ml_home_probability"] = home_probs
                df["ml_pick"] = np.where(
                    df["ml_home_probability"] >= 0.5,
                    df["home_team"],
                    df["away_team"],
                )
                df["ml_win_probability"] = np.where(
                    df["ml_home_probability"] >= 0.5,
                    df["ml_home_probability"],
                    1.0 - df["ml_home_probability"],
                )

                # V3.1 final pick is the optimized production model.
                df["final_pick"] = df["ml_pick"]
                df["final_probability"] = df["ml_win_probability"]
                df["agreement"] = np.where(df["v3_pick"] == df["ml_pick"], "YES", "NO")
                df["confidence_level_v31"] = df["ml_win_probability"].map(_confidence)

                display = df[[
                    "rank",
                    "away_team",
                    "home_team",
                    "v3_pick",
                    "ml_pick",
                    "final_pick",
                    "final_probability",
                    "confidence_level_v31",
                    "agreement",
                    "projected_score",
                ]].copy()

                display["final_probability"] = (
                    display["final_probability"] * 100
                ).round(1).astype(str) + "%"

                display = display.rename(columns={
                    "away_team": "Away",
                    "home_team": "Home",
                    "v3_pick": "Original V3",
                    "ml_pick": "V3.1 ML",
                    "final_pick": "Final V3.1 Pick",
                    "final_probability": "Win %",
                    "confidence_level_v31": "Confidence",
                    "agreement": "Models Agree",
                    "projected_score": "V3 Projected Score",
                })

                st.dataframe(display, width="stretch", hide_index=True)

                disagreements = df[df["agreement"].eq("NO")]
                if len(disagreements):
                    st.subheader("Model Disagreements")
                    st.write(
                        "These games deserve extra review because the original V3 and "
                        "optimized V3.1 model disagree."
                    )
                    st.dataframe(
                        disagreements[[
                            "away_team","home_team","v3_pick","ml_pick",
                            "ml_win_probability"
                        ]],
                        width="stretch",
                        hide_index=True
                    )
            else:
                st.dataframe(df, width="stretch", hide_index=True)

        except Exception as e:
            st.error("V3.1 weekly picks failed")
            st.code(str(e))

with tabs[1]:
    st.subheader("Original V3 Week Backtest")
    c1, c2 = st.columns(2)
    s = c1.number_input("Season", 2020, 2025, 2025, key="wbs")
    w = c2.number_input("Week", 1, 18, 1, key="wbw")
    if st.button("Run Week Backtest"):
        try:
            d = backtest(int(s))
            d = d[d["week"].eq(int(w))]
            total = len(d)
            correct = int(d["correct"].sum()) if total else 0
            st.metric("Accuracy", f"{(100*correct/total if total else 0):.1f}%")
            st.dataframe(d, width="stretch", hide_index=True)
        except Exception as e:
            st.error("Week backtest failed")
            st.code(str(e))

with tabs[2]:
    st.subheader("Original V3 Season Backtest")
    s = st.number_input("Season", 2020, 2025, 2025, key="sbs")
    if st.button("Run Season Backtest"):
        try:
            d = backtest(int(s))
            total = len(d)
            correct = int(d["correct"].sum()) if total else 0
            st.metric("Season Accuracy", f"{(100*correct/total if total else 0):.1f}%")
            weekly = d.groupby("week", as_index=False).agg(
                Games=("correct","size"),
                Correct=("correct","sum"),
                Accuracy=("correct","mean"),
            )
            weekly["Wrong"] = weekly["Games"] - weekly["Correct"]
            weekly["Accuracy"] = (weekly["Accuracy"] * 100).round(1)
            st.dataframe(weekly, width="stretch", hide_index=True)
        except Exception as e:
            st.error("Season backtest failed")
            st.code(str(e))

with tabs[3]:
    st.subheader("V3.1 Optimization Results")

    summary_path = ROOT / "outputs" / "v31_summary.json"
    trials_path = ROOT / "outputs" / "v31_validation_trials.csv"
    holdout_path = ROOT / "outputs" / "v31_2025_holdout_games.csv"
    weekly_path = ROOT / "outputs" / "v31_2025_weekly_results.csv"
    weights_path = ROOT / "outputs" / "v31_learned_weights.csv"

    if not all(p.exists() for p in [
        summary_path, trials_path, holdout_path, weekly_path, weights_path, MODEL_PATH
    ]):
        st.warning(
            "No V3.1 model files found yet. Run the GitHub Action "
            "'Train NFL Picker V3.1' and refresh this page."
        )
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        trials = pd.read_csv(trials_path)
        holdout = pd.read_csv(holdout_path)
        weekly = pd.read_csv(weekly_path)
        weights = pd.read_csv(weights_path)

        hm = summary["holdout_2025"]

        a, b, c = st.columns(3)
        a.metric("Selected C", summary["selected_C"])
        b.metric("2025 Games", int(hm["games"]))
        c.metric("2025 Holdout Accuracy", f"{float(hm['accuracy'])*100:.1f}%")

        st.subheader("2024 Validation Trials")
        st.dataframe(trials, width="stretch", hide_index=True)

        st.subheader("Learned Production Feature Importance")
        ww = weights.copy()
        ww["importance"] = (ww["importance"] * 100).round(1)
        st.dataframe(ww, width="stretch", hide_index=True)

        st.subheader("2025 Week-by-Week Holdout")
        st.dataframe(weekly, width="stretch", hide_index=True)

        st.subheader("Every 2025 V3.1 Holdout Pick")
        keep = [
            c for c in [
                "week","away_team","home_team","ml_pick",
                "ml_win_probability","actual_winner","correct"
            ] if c in holdout.columns
        ]
        st.dataframe(holdout[keep], width="stretch", hide_index=True)
        with tabs[4]:
            render_pick_history(root)
