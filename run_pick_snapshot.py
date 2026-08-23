from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))
from nfl_picker_v3.pick_history import generate
p=argparse.ArgumentParser()
p.add_argument("--season",type=int,required=True)
p.add_argument("--week",type=int,required=True)
p.add_argument("--stage",required=True)
a=p.parse_args()
print(generate(ROOT,a.season,a.week,a.stage))
