from pathlib import Path
import streamlit as st
from nfl_picker_v3.pick_history import compare

def render_pick_history(root: Path):
    st.subheader("EARLY → UPDATED → FINAL Pick Tracker")
    c1,c2=st.columns(2)
    season=c1.number_input("History season",2020,2035,2026,key="hist_season")
    week=c2.number_input("History week",1,18,1,key="hist_week")
    df=compare(root,int(season),int(week))
    if df.empty:
        st.info("No saved stage picks yet. Run the GitHub Action 'Save NFL Picker V3.1 Stage'.")
        return
    changed=df[df["Status"].eq("⚠ PICK CHANGED")]
    if len(changed):
        st.warning(f"{len(changed)} game(s) changed predicted winner.")
    st.dataframe(df,width="stretch",hide_index=True)
    for _,r in changed.iterrows():
        st.error(f"{r['away_team']} @ {r['home_team']}: {r['Pick_History']}")
