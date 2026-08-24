from __future__ import annotations

import pandas as pd


def _normalize_pick(value):
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    return text if text else None


def agreement_result(original_v3_pick, v31_ml_pick, optional_third_pick=None) -> dict:
    picks = [
        _normalize_pick(original_v3_pick),
        _normalize_pick(v31_ml_pick),
        _normalize_pick(optional_third_pick),
    ]
    picks = [p for p in picks if p is not None]

    if not picks:
        return {
            "agreement_count": 0,
            "agreeing_on": None,
            "agreement_level": "NONE",
            "models_agree": False,
        }

    counts = pd.Series(picks).value_counts()
    top_team = counts.index[0]
    top_count = int(counts.iloc[0])
    total = len(picks)

    if total == 1:
        level = "NONE"
        agree = False
    elif total == 2:
        if top_count == 2:
            level = "AGREE"
            agree = True
        else:
            level = "SPLIT"
            agree = False
    else:
        if top_count == 3:
            level = "STRONG_AGREE"
            agree = True
        elif top_count == 2:
            level = "AGREE"
            agree = True
        else:
            level = "SPLIT"
            agree = False

    return {
        "agreement_count": total,
        "agreeing_on": top_team if agree else None,
        "agreement_level": level,
        "models_agree": agree,
    }


def apply_model_agreement(
    df: pd.DataFrame,
    original_col: str = "v3_pick",
    ml_col: str = "ml_pick",
    third_col: str | None = None,
) -> pd.DataFrame:
    out = df.copy()

    if original_col not in out.columns:
        raise ValueError(f"Missing original model pick column: {original_col}")
    if ml_col not in out.columns:
        raise ValueError(f"Missing ML pick column: {ml_col}")
    if third_col and third_col not in out.columns:
        raise ValueError(f"Missing third pick column: {third_col}")

    results = []
    for _, row in out.iterrows():
        third = row[third_col] if third_col else None
        results.append(
            agreement_result(
                row[original_col],
                row[ml_col],
                third,
            )
        )

    result_df = pd.DataFrame(results, index=out.index)
    for col in result_df.columns:
        out[col] = result_df[col]

    return out
