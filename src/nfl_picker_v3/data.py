
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import nflreadpy as nfl


def _to_pandas(df):
    return df.to_pandas() if hasattr(df, "to_pandas") else pd.DataFrame(df)


@dataclass
class DataBundle:
    schedules: pd.DataFrame
    player_stats: pd.DataFrame
    pbp: pd.DataFrame


def _load_checked(loader, seasons, label, **kwargs):
    seasons = sorted(set(int(s) for s in seasons))

    try:
        df = _to_pandas(loader(seasons, **kwargs))
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load {label} for seasons {seasons}. "
            "No seasons were silently removed."
        ) from exc

    if df.empty:
        raise ValueError(f"{label} returned no records.")

    if "season" not in df.columns:
        raise ValueError(f"{label} is missing its season column.")

    available = set(
        pd.to_numeric(df["season"], errors="coerce").dropna().astype(int)
    )
    missing = set(seasons) - available

    if missing:
        raise ValueError(
            f"{label} is missing requested seasons: {sorted(missing)}"
        )

    return df


def load_bundle(seasons):
    seasons = sorted(set(int(s) for s in seasons))

    schedules = _load_checked(
        nfl.load_schedules, seasons, "Schedules"
    )

    player_stats = _load_checked(
        nfl.load_player_stats,
        seasons,
        "Weekly player statistics",
        summary_level="week",
    )

    pbp = _load_checked(
        nfl.load_pbp, seasons, "Play-by-play statistics"
    )

    return DataBundle(schedules, player_stats, pbp)


def load_injury_overrides():
    path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "injury_adjustments.csv"
    )

    if not path.exists():
        return pd.DataFrame(
            columns=["season", "week", "team", "adjustment", "notes"]
        )

    return pd.read_csv(path)
