import numpy as np,pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,brier_score_loss,log_loss,roc_auc_score
from .engine import build_context
from .model import predict_game
FEATURES=["edge_qb","edge_offense","edge_defense","edge_trenches","edge_injuries","edge_market","edge_turnovers","edge_form","edge_situational","edge_matchup","edge_coaching"]
def fit_holdout_model():
    s,tw,qw,inj=build_context(list(range(2018,2026))); games=s[s.season.between(2020,2025)&s.home_score.notna()&s.away_score.notna()]; rows=[]
    for _,g in games.sort_values(["season","week","gameday"]).iterrows():
        if g.home_score==g.away_score: continue
        p=predict_game(g,s,tw,qw,inj); r={"season":int(g.season),"week":int(g.week),"away_team":g.away_team,"home_team":g.home_team,"home_win":int(g.home_score>g.away_score)}; [r.__setitem__(f,float(p.get(f,0))) for f in FEATURES]; rows.append(r)
    df=pd.DataFrame(rows); tr=df[df.season.between(2020,2024)].copy(); te=df[df.season.eq(2025)].copy(); Xtr=tr[FEATURES].fillna(0); ytr=tr.home_win.astype(int); Xte=te[FEATURES].fillna(0); yte=te.home_win.astype(int); pipe=Pipeline([("scale",StandardScaler()),("clf",LogisticRegression(max_iter=3000,random_state=42))]); pipe.fit(Xtr,ytr); hp=pipe.predict_proba(Xte)[:,1]; te["home_win_probability_ml"]=hp; te["ml_pick"]=np.where(hp>=.5,te.home_team,te.away_team); te["actual_winner"]=np.where(te.home_win.eq(1),te.home_team,te.away_team); te["correct"]=(te.ml_pick==te.actual_winner).astype(int); te["ml_win_probability"]=np.where(hp>=.5,hp,1-hp); co=pd.Series(pipe.named_steps["clf"].coef_[0],index=FEATURES); im=co.abs()/co.abs().sum(); weights=pd.DataFrame({"feature":FEATURES,"coefficient":co.values,"importance":im.values}).sort_values("importance",ascending=False); weekly=te.groupby("week",as_index=False).agg(Games=("correct","size"),Correct=("correct","sum"),Accuracy=("correct","mean")); weekly["Wrong"]=weekly.Games-weekly.Correct; weekly["Accuracy"]=(weekly.Accuracy*100).round(1); metrics={"train_games":len(tr),"holdout_games":len(te),"holdout_correct":int(te.correct.sum()),"holdout_accuracy":accuracy_score(yte,hp>=.5),"brier_score":brier_score_loss(yte,hp),"log_loss":log_loss(yte,hp),"roc_auc":roc_auc_score(yte,hp)}; return pipe,tr,te,metrics,weights,weekly
