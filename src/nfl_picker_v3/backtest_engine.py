import pandas as pd
from .engine import build_context
from .model import predict_game
def backtest(season):
    s,tw,qw,inj=build_context([season-2,season-1,season]); games=s[(s.season==season)&s.home_score.notna()&s.away_score.notna()]; rows=[]
    for _,g in games.sort_values(["week","gameday"]).iterrows():
        if g.home_score==g.away_score: continue
        p=predict_game(g,s,tw,qw,inj); actual=g.home_team if g.home_score>g.away_score else g.away_team; p["actual_winner"]=actual; p["actual_score"]=f"{g.away_team} {int(g.away_score)} - {g.home_team} {int(g.home_score)}"; p["correct"]=int(p["pick"]==actual); rows.append(p)
    return pd.DataFrame(rows)
