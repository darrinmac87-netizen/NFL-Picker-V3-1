from pathlib import Path
import json,math,pandas as pd
from .features import snap,rest_days
def cfg(): return json.loads((Path(__file__).resolve().parents[2]/"config.json").read_text())
def sig(x): return 1/(1+math.exp(-max(min(x,20),-20)))
def cl(p): return 5 if p>=.75 else 4 if p>=.67 else 3 if p>=.60 else 2 if p>=.55 else 1
def diff(a,b,s):
    if pd.isna(a) and pd.isna(b): return 0.0
    if pd.isna(a): a=b
    if pd.isna(b): b=a
    return (float(a)-float(b))/s
def predict_game(g,s,tw,qw,inj):
    C=cfg(); w=C["weights"]; season=int(g.season); week=int(g.week); home=g.home_team; away=g.away_team; hs=snap(tw,qw,season,week,home,C.get("recent_games",4)); a=snap(tw,qw,season,week,away,C.get("recent_games",4)); c={}
    c["qb"]=diff(hs["qb_eff"],a["qb_eff"],2); c["offense"]=diff(hs["off_epa"],a["off_epa"],.1); c["defense"]=diff(a["def_epa_allowed"],hs["def_epa_allowed"],.1); c["trenches"]=(diff(hs["off_success"],a["off_success"],.05)+diff(a["def_success_allowed"],hs["def_success_allowed"],.05))/2; c["injuries"]=0.0; c["market"]=0.0; c["turnovers"]=c["qb"]*.3; c["form"]=diff(hs["form_off_epa"],a["form_off_epa"],.1); c["situational"]=.75+(rest_days(s,home,season,week,g.gameday)-rest_days(s,away,season,week,g.gameday))/7; hm=(0 if pd.isna(hs["off_epa"]) else hs["off_epa"])-(0 if pd.isna(a["def_epa_allowed"]) else a["def_epa_allowed"]); am=(0 if pd.isna(a["off_epa"]) else a["off_epa"])-(0 if pd.isna(hs["def_epa_allowed"]) else hs["def_epa_allowed"]); c["matchup"]=(hm-am)/.1; c["coaching"]=0.0
    raw=C.get("home_field_logit",.12)+sum(w.get(k,0)*max(min(v,3),-3) for k,v in c.items()); hp=sig(raw); pick=home if hp>=.5 else away; wp=max(hp,1-hp); margin=max(-17,min(17,(hp-.5)*34)); h=round(22+margin/2); aa=round(22-margin/2)
    return {"season":season,"week":week,"away_team":away,"home_team":home,"pick":pick,"home_win_prob":hp,"win_probability":wp,"confidence_level":cl(wp),"projected_score":f"{away} {aa} - {home} {h}","upset_risk":"Low" if wp>=.72 else ("Medium" if wp>=.60 else "High"),"do_not_lock":wp<.60,**{f"edge_{k}":float(v) for k,v in c.items()}}
