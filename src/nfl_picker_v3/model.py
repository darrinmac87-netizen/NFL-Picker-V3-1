from pathlib import Path
import json,math,pandas as pd
from .features import snap,rest_days
def cfg(): return json.loads((Path(__file__).resolve().parents[2]/"config.json").read_text())
def sig(x): return 1/(1+math.exp(-max(min(x,20),-20)))
def cl(p): return 5 if p>=.75 else 4 if p>=.67 else 3 if p>=.60 else 2 if p>=.55 else 1
def diff(a,b,s):
    if pd.isna(a) and pd.isna(b): return 0.0
from pathlib import Path
import json
import math

import pandas as pd

from .features import snap, rest_days


def cfg():
    path = Path(__file__).resolve().parents[2] / "config.json"
    return json.loads(path.read_text())


def sig(x):
    return 1 / (1 + math.exp(-max(min(x, 20), -20)))


def cl(p):
    return 5 if p >= .75 else 4 if p >= .67 else 3 if p >= .60 else 2 if p >= .55 else 1


def diff(a, b, scale):
    if pd.isna(a) and pd.isna(b):
        return 0.0
    if pd.isna(a):
        a = b
    if pd.isna(b):
        b = a
    return (float(a) - float(b)) / scale


def predict_game(g, s, tw, qw, inj):
    C = cfg()
    w = C["weights"]
    season = int(g.season)
    week = int(g.week)
    home = g.home_team
    away = g.away_team

    hs = snap(tw, qw, season, week, home, C.get("recent_games", 4))
    a = snap(tw, qw, season, week, away, C.get("recent_games", 4))

    c = {}

    # Preserve the V4.0 baseline features.
    c["qb"] = diff(hs["qb_eff"], a["qb_eff"], 2)
    c["offense"] = diff(hs["off_epa"], a["off_epa"], .1)
    c["defense"] = diff(a["def_epa_allowed"], hs["def_epa_allowed"], .1)
    c["trenches"] = (
        diff(hs["off_success"], a["off_success"], .05)
        + diff(a["def_success_allowed"], hs["def_success_allowed"], .05)
    ) / 2

    c["injuries"] = 0.0
    c["market"] = 0.0
    c["turnovers"] = c["qb"] * .3
    c["form"] = diff(hs["form_off_epa"], a["form_off_epa"], .1)

    c["situational"] = .75 + (
        rest_days(s, home, season, week, g.gameday)
        - rest_days(s, away, season, week, g.gameday)
    ) / 7

    hm = (
        (0 if pd.isna(hs["off_epa"]) else hs["off_epa"])
        - (0 if pd.isna(a["def_epa_allowed"]) else a["def_epa_allowed"])
    )
    am = (
        (0 if pd.isna(a["off_epa"]) else a["off_epa"])
        - (0 if pd.isna(hs["def_epa_allowed"]) else hs["def_epa_allowed"])
    )

    c["matchup"] = (hm - am) / .1
    c["coaching"] = 0.0

    # New V4.1 statistical features.
    # Positive values favor the home team.
    c["pass_protection"] = diff(
        a.get("sack_allowed_rate"),
        hs.get("sack_allowed_rate"),
        .03
    )
    c["pass_rush"] = diff(
        hs.get("sack_generated_rate"),
        a.get("sack_generated_rate"),
        .03
    )
    c["qb_hit_protection"] = diff(
        a.get("qb_hit_allowed_rate"),
        hs.get("qb_hit_allowed_rate"),
        .05
    )
    c["qb_hit_pressure"] = diff(
        hs.get("qb_hit_generated_rate"),
        a.get("qb_hit_generated_rate"),
        .05
    )
    c["rushing"] = diff(
        hs.get("run_success"),
        a.get("run_success"),
        .05
    )
    c["run_defense"] = diff(
        a.get("run_success_allowed"),
        hs.get("run_success_allowed"),
        .05
    )
    c["turnover_protection"] = diff(
        a.get("turnover_rate"),
        hs.get("turnover_rate"),
        .02
    )
    c["takeaways"] = diff(
        hs.get("turnover_forced_rate"),
        a.get("turnover_forced_rate"),
        .02
    )

    # Candidate features are recorded, but have zero weight
    # until independently validated.
    candidate_features = {
        "pass_protection",
        "pass_rush",
        "qb_hit_protection",
        "qb_hit_pressure",
        "rushing",
        "run_defense",
        "turnover_protection",
        "takeaways",
    }

    raw = C.get("home_field_logit", .12) + sum(
        (
            0.0 if k in candidate_features
            else w.get(k, 0.0)
        ) * max(min(v, 3), -3)
        for k, v in c.items()
    )

    hp = sig(raw)
    pick = home if hp >= .5 else away
    wp = max(hp, 1 - hp)
    margin = max(-17, min(17, (hp - .5) * 34))
    h = round(22 + margin / 2)
    aa = round(22 - margin / 2)

    return {
        "season": season,
        "week": week,
        "away_team": away,
        "home_team": home,
        "pick": pick,
        "home_win_prob": hp,
        "win_probability": wp,
        "confidence_level": cl(wp),
        "projected_score": f"{away} {aa} - {home} {h}",
        "upset_risk": (
            "Low" if wp >= .72
            else "Medium" if wp >= .60
            else "High"
        ),
        "do_not_lock": wp < .60,
        **{f"edge_{k}": float(v) for k, v in c.items()},
    }

    if pd.isna(a): a=b
    if pd.isna(b): b=a
    return (float(a)-float(b))/s
def predict_game(g,s,tw,qw,inj):
    C=cfg(); w=C["weights"]; season=int(g.season); week=int(g.week); home=g.home_team; away=g.away_team; hs=snap(tw,qw,season,week,home,C.get("recent_games",4)); a=snap(tw,qw,season,week,away,C.get("recent_games",4)); c={}
    c["qb"]=diff(hs["qb_eff"],a["qb_eff"],2); c["offense"]=diff(hs["off_epa"],a["off_epa"],.1); c["defense"]=diff(a["def_epa_allowed"],hs["def_epa_allowed"],.1); c["trenches"]=(diff(hs["off_success"],a["off_success"],.05)+diff(a["def_success_allowed"],hs["def_success_allowed"],.05))/2; c["injuries"]=0.0; c["market"]=0.0; c["turnovers"]=c["qb"]*.3; c["form"]=diff(hs["form_off_epa"],a["form_off_epa"],.1); c["situational"]=.75+(rest_days(s,home,season,week,g.gameday)-rest_days(s,away,season,week,g.gameday))/7; hm=(0 if pd.isna(hs["off_epa"]) else hs["off_epa"])-(0 if pd.isna(a["def_epa_allowed"]) else a["def_epa_allowed"]); am=(0 if pd.isna(a["off_epa"]) else a["off_epa"])-(0 if pd.isna(hs["def_epa_allowed"]) else hs["def_epa_allowed"]); c["matchup"]=(hm-am)/.1; c["coaching"]=0.0
    raw=C.get("home_field_logit",.12)+sum(w.get(k,0)*max(min(v,3),-3) for k,v in c.items()); hp=sig(raw); pick=home if hp>=.5 else away; wp=max(hp,1-hp); margin=max(-17,min(17,(hp-.5)*34)); h=round(22+margin/2); aa=round(22-margin/2)
    return {"season":season,"week":week,"away_team":away,"home_team":home,"pick":pick,"home_win_prob":hp,"win_probability":wp,"confidence_level":cl(wp),"projected_score":f"{away} {aa} - {home} {h}","upset_risk":"Low" if wp>=.72 else ("Medium" if wp>=.60 else "High"),"do_not_lock":wp<.60,**{f"edge_{k}":float(v) for k,v in c.items()}}
