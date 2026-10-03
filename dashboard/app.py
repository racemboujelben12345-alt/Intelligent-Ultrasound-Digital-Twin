# streamlit run dashboard/app.py   (après python main.py)
import streamlit as st, pandas as pd
from pathlib import Path
O = Path(__file__).resolve().parent.parent / "outputs"
st.title("SCAN A Digital Twin V2"); st.markdown((O/"report.md").read_text())
for f in ["fig1_drift_timeline", "fig2_health_state", "fig3_fault_lab", "fig4_virtual_faults"]: st.image(str(O/f"{f}.png"))
st.dataframe(pd.read_csv(O/"features.csv"))
