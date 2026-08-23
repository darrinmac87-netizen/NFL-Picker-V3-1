import numpy as np
import pandas as pd

ALIASES = {"JAC":"JAX","LA":"LAR","STL":"LAR","SD":"LAC","OAK":"LV"}

def norm(x):
    return ALIASES.get(str(x), str(x)) if pd.notna(x) else x

def prepare_schedule(df):
    df = df.copy()
    if "game_type" in df.columns:
        df = df[df["game_type"].eq("REG")].copy()
    for c in ["home_team", "away_team"]:
        if c in df.columns:
            df[c] = df[c].map(norm)
    if "gameday" in df.columns:
        df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    return df

def team_pbp_weekly(pbp):
    if pbp is None or len(pbp) == 0:
        return pd.DataFrame(columns=[
            "season","week","team","off_epa","off_success",
            "def_epa_allowed","def_success_allowed"
        ])

    df = pbp.copy()
    if "season_type" in df.columns:
        df = df[df["season_type"].eq("REG")]
    if "play_type" in df.columns:
        df = df[df["play_type"].isin(["pass","run"])]

    required = {"posteam","defteam","epa","season","week"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Play-by-play data is missing required columns: {sorted(missing)}")

    df = df[df["posteam"].notna() & df["defteam"].notna()].copy()
    df["posteam"] = df["posteam"].map(norm)
    df["defteam"] = df["defteam"].map(norm)
    df["epa"] = pd.to_numeric(df["epa"], errors="coerce")
    df["success"] = (df["epa"] > 0).astype(float)

    off = (
        df.groupby(["season","week","posteam"], as_index=False)
        .agg(off_epa=("epa","mean"), off_success=("success","mean"))
        .rename(columns={"posteam":"team"})
    )
    de = (
        df.groupby(["season","week","defteam"], as_index=False)
        .agg(def_epa_allowed=("epa","mean"),
             def_success_allowed=("success","mean"))
        .rename(columns={"defteam":"team"})
    )
    return off.merge(de, on=["season","week","team"], how="outer")

def _numeric_series(df, candidates, default=0.0):
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")

def quarterback_weekly(ps):
    if ps is None or len(ps) == 0:
        return pd.DataFrame(columns=["season","week","team","qb_eff"])

    df = ps.copy()

    if "season_type" in df.columns:
        df = df[df["season_type"].eq("REG")].copy()
    if "position" in df.columns:
        df = df[df["position"].eq("QB")].copy()

    team_col = (
        "recent_team" if "recent_team" in df.columns
        else "team" if "team" in df.columns
        else None
    )
    if team_col is None or len(df) == 0:
        return pd.DataFrame(columns=["season","week","team","qb_eff"])

    required = {"season","week"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Player stats are missing required columns: {sorted(missing)}")

    df["team"] = df[team_col].map(norm)

    att = _numeric_series(df, ["attempts", "passing_attempts"])
    yards = _numeric_series(df, ["passing_yards"])
    td = _numeric_series(df, ["passing_tds", "passing_touchdowns"])
    interceptions = _numeric_series(df, ["interceptions", "passing_interceptions"])
    sacks = _numeric_series(df, ["sacks_suffered", "sacks"])

    att_safe = att.replace(0, np.nan)
    dropbacks_safe = (att + sacks).replace(0, np.nan)

    df["qb_eff"] = (
        yards / att_safe
        + 15.0 * (td / att_safe)
        - 20.0 * (interceptions / att_safe)
        - 8.0 * (sacks / dropbacks_safe)
    ).replace([np.inf, -np.inf], np.nan)

    df["att"] = att

    idx = (
        df.groupby(["season","week","team"])["att"]
        .idxmax()
        .dropna()
        .astype(int)
    )
    if len(idx) == 0:
        return pd.DataFrame(columns=["season","week","team","qb_eff"])

    return df.loc[idx, ["season","week","team","qb_eff"]].reset_index(drop=True)

def snap(tw, qw, season, week, team, recent=4):
    h = tw[
        ((tw.season < season) | ((tw.season == season) & (tw.week < week)))
        & tw.team.eq(team)
    ].sort_values(["season","week"])

    cur = h[h.season.eq(season)]
    prev = h[h.season.eq(season - 1)]
    cw, pw = (.35,.65) if week <= 3 else ((.70,.30) if week <= 8 else (.85,.15))

    out = {}
    for m in ["off_epa","off_success","def_epa_allowed","def_success_allowed"]:
        vals, ws = [], []
        a = cur[m].tail(recent).mean() if m in cur.columns else np.nan
        b = prev[m].tail(8).mean() if m in prev.columns else np.nan
        if pd.notna(a):
            vals.append(a); ws.append(cw)
        if pd.notna(b):
            vals.append(b); ws.append(pw)
        out[m] = np.average(vals, weights=ws) if vals else np.nan

    q = qw[
        ((qw.season < season) | ((qw.season == season) & (qw.week < week)))
        & qw.team.eq(team)
    ].sort_values(["season","week"])

    qcur = q[q.season.eq(season)]
    qprev = q[q.season.eq(season - 1)]
    a = qcur["qb_eff"].tail(recent).mean() if "qb_eff" in qcur.columns else np.nan
    b = qprev["qb_eff"].tail(8).mean() if "qb_eff" in qprev.columns else np.nan

    vals, ws = [], []
    if pd.notna(a):
        vals.append(a); ws.append(cw)
    if pd.notna(b):
        vals.append(b); ws.append(pw)

    out["qb_eff"] = np.average(vals, weights=ws) if vals else np.nan
    out["form_off_epa"] = (
        cur["off_epa"].tail(recent).mean()
        if len(cur) and "off_epa" in cur.columns
        else np.nan
    )
    return out

def rest_days(s, team, season, week, gday):
    p = s[
        (s.season == season)
        & (s.week < week)
        & ((s.home_team == team) | (s.away_team == team))
        & s.gameday.notna()
    ]
    return 7.0 if p.empty or pd.isna(gday) else max(
        0, (gday - p.gameday.max()).days
    )
