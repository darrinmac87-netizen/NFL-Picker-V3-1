from pathlib import Path
from datetime import datetime, timezone
import json, math
import numpy as np
import pandas as pd
from .engine import weekly_predictions

STAGES=("EARLY","UPDATED","FINAL")

def sig(x):
    x=max(min(float(x),20),-20)
    return 1/(1+math.exp(-x))

def conf(p):
    p=max(float(p),1-float(p))
    return 5 if p>=.75 else 4 if p>=.67 else 3 if p>=.60 else 2 if p>=.55 else 1

def load_model(root):
    p=Path(root)/"outputs"/"v31_production_model.json"
    if not p.exists():
        raise FileNotFoundError("Run Train NFL Picker V3.1 first.")
    return json.loads(p.read_text())

def ml_home(row,m):
    vals=np.array([0.0 if pd.isna(row.get(f,0.0)) else float(row.get(f,0.0)) for f in m["features"]])
    mean=np.array(m["scaler_mean"],float); scale=np.array(m["scaler_scale"],float)
    coef=np.array(m["coefficients"],float); scale=np.where(scale==0,1.0,scale)
    return sig(float(m["intercept"])+float(np.dot(coef,(vals-mean)/scale)))

def generate(root,season,week,stage):
    root=Path(root); stage=stage.upper()
    if stage not in STAGES: raise ValueError("stage must be EARLY, UPDATED, or FINAL")
    m=load_model(root)
    df=weekly_predictions(int(season),int(week)).copy()
    hp=df.apply(lambda r: ml_home(r,m),axis=1)
    df["v31_pick"]=np.where(hp>=.5,df["home_team"],df["away_team"])
    df["v31_win_probability"]=np.where(hp>=.5,hp,1-hp)
    df["v31_confidence"]=df["v31_win_probability"].map(conf)
    df["original_v3_pick"]=df["pick"]
    df["stage"]=stage
    df["generated_at_utc"]=datetime.now(timezone.utc).isoformat()
    keep=["season","week","away_team","home_team","original_v3_pick","v31_pick",
          "v31_win_probability","v31_confidence","projected_score","upset_risk",
          "stage","generated_at_utc"]
    out=root/"outputs"/"pick_history"/str(int(season))/f"week_{int(week):02d}"
    out.mkdir(parents=True,exist_ok=True)
    p=out/f"{stage.lower()}.csv"
    df[keep].to_csv(p,index=False)
    return p

def compare(root,season,week):
    folder=Path(root)/"outputs"/"pick_history"/str(int(season))/f"week_{int(week):02d}"
    frames={}
    for s in STAGES:
        p=folder/f"{s.lower()}.csv"
        if p.exists(): frames[s]=pd.read_csv(p)
    if not frames: return pd.DataFrame()
    first=frames[next(iter(frames))]
    out=first[["away_team","home_team"]].drop_duplicates().copy()
    for s in STAGES:
        if s in frames:
            x=frames[s][["away_team","home_team","v31_pick","v31_win_probability","v31_confidence"]].copy()
            x=x.rename(columns={"v31_pick":f"{s}_Pick","v31_win_probability":f"{s}_Win_Prob","v31_confidence":f"{s}_Confidence"})
            out=out.merge(x,on=["away_team","home_team"],how="left")
        else:
            out[f"{s}_Pick"]=pd.NA
    def status(r):
        picks=[r.get(f"{s}_Pick") for s in STAGES if pd.notna(r.get(f"{s}_Pick"))]
        return "⚠ PICK CHANGED" if len(set(map(str,picks)))>1 else "Stable"
    def hist(r):
        return " → ".join(f"{s}: {r.get(f'{s}_Pick')}" for s in STAGES if pd.notna(r.get(f"{s}_Pick")))
    out["Status"]=out.apply(status,axis=1)
    out["Pick_History"]=out.apply(hist,axis=1)
    return out
