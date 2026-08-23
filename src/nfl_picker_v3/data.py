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

def _load_with_fallback(loader,seasons,*args,**kwargs):
    seasons=sorted(set(int(s) for s in seasons)); last=None
    for end in range(len(seasons),0,-1):
        try: return _to_pandas(loader(seasons[:end],*args,**kwargs))
        except Exception as exc: last=exc
    if last: raise last
    return pd.DataFrame()

def load_bundle(seasons):
    schedules=_to_pandas(nfl.load_schedules(seasons))
    player_stats=_load_with_fallback(nfl.load_player_stats,seasons,summary_level="week")
    pbp=_load_with_fallback(nfl.load_pbp,seasons)
    return DataBundle(schedules,player_stats,pbp)

def load_injury_overrides():
    path=Path(__file__).resolve().parents[2]/"data"/"injury_adjustments.csv"
    if not path.exists(): return pd.DataFrame(columns=["season","week","team","adjustment","notes"])
    return pd.read_csv(path)
