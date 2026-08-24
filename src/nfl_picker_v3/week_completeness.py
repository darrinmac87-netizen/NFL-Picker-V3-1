from __future__ import annotations

import pandas as pd


# Expected number of regular-season games by week for the 2026 NFL season.
# These total 272 games.
EXPECTED_2026_GAMES = {
    1: 16,
    2: 16,
    3: 16,
    4: 16,
    5: 15,
    6: 14,
    7: 14,
    8: 14,
    9: 15,
    10: 14,
    11: 13,
    12: 16,
    13: 14,
    14: 15,
    15: 16,
    16: 16,
    17: 16,
    18: 16,
}


def expected_games(season: int, week: int) -> int | None:
    """
    Return the expected number of regular-season games for a week.

    Currently validated for the 2026 season.
    """
    season = int(season)
    week = int(week)

    if season == 2026:
        return EXPECTED_2026_GAMES.get(week)

    return None


def _unique_game_count(df: pd.DataFrame) -> int:
    """
    Count unique away/home matchups in a weekly predictions dataframe.
    """
    if df.empty:
        return 0

    required = {"away_team", "home_team"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Week completeness check requires columns: "
            "away_team and home_team."
        )

    games = (
        df[["away_team", "home_team"]]
        .dropna()
        .drop_duplicates()
    )

    return int(len(games))


def duplicate_games(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return any duplicate away/home matchups.
    """
    if df.empty:
        return pd.DataFrame(columns=["away_team", "home_team"])

    required = {"away_team", "home_team"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["away_team", "home_team"])

    mask = df.duplicated(
        subset=["away_team", "home_team"],
        keep=False,
    )

    return (
        df.loc[mask, ["away_team", "home_team"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def check_week_completeness(
    df: pd.DataFrame,
    season: int,
    week: int,
) -> dict:
    """
    Compare V3.1's weekly prediction count with the expected NFL game count.

    Returns:
      expected_games
      found_games
      missing_games
      complete
      duplicate_count
      status
      message
    """
    season = int(season)
    week = int(week)

    expected = expected_games(season, week)
    found = _unique_game_count(df)
    duplicates = duplicate_games(df)

    if expected is None:
        return {
            "season": season,
            "week": week,
            "expected_games": None,
            "found_games": found,
            "missing_games": None,
            "complete": None,
            "duplicate_count": int(len(duplicates)),
            "status": "UNKNOWN",
            "message": (
                f"{found} unique game(s) found. "
                "No expected game-count table is installed for "
                f"{season} yet."
            ),
        }

    missing = max(expected - found, 0)
    complete = found == expected and len(duplicates) == 0

    if complete:
        status = "COMPLETE"
        message = (
            f"✅ Week {week} schedule complete: "
            f"{found} of {expected} games found."
        )
    elif found < expected:
        status = "INCOMPLETE"
        message = (
            f"⚠ Week {week} schedule incomplete: "
            f"{found} of {expected} games found. "
            f"{missing} game(s) missing."
        )
    elif found > expected:
        status = "TOO_MANY"
        message = (
            f"⚠ Week {week} has too many unique games: "
            f"{found} found, expected {expected}."
        )
    else:
        status = "DUPLICATES"
        message = (
            f"⚠ Week {week} has duplicate matchup rows. "
            f"{found} unique games found."
        )

    return {
        "season": season,
        "week": week,
        "expected_games": expected,
        "found_games": found,
        "missing_games": missing,
        "complete": complete,
        "duplicate_count": int(len(duplicates)),
        "status": status,
        "message": message,
    }
