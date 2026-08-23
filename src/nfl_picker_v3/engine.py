import pandas as pd
from .data import load_bundle,load_injury_overrides
from .features import prepare_schedule,team_pbp_weekly,quarterback_weekly
from .model import predict_game
def build_context(seasons):
    b=load_bundle(seasons); s=prepare_schedule(b.schedules); return s,team_pbp_weekly(b.pbp),quarterback_weekly(b.player_stats),load_injury_overrides()
def weekly_predictions(season,week):
    s,tw,qw,inj=build_context([season-2,season-1,season]); games=s[(s.season==season)&(s.week==week)].copy();
    if games.empty: raise ValueError(f"No regular-season games found for {season} Week {week}.")
    out=pd.DataFrame([predict_game(g,s,tw,qw,inj) for _,g in games.iterrows()]).sort_values(["confidence_level","win_probability"],ascending=[False,False]).reset_index(drop=True); out.insert(0,"rank",range(1,len(out)+1)); return out
