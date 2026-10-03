"""SCAN A Digital Twin — engineering dashboard.
Presentation layer only: reads pipeline artifacts from outputs/.
"""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

st.set_page_config(page_title="SCAN A Digital Twin", page_icon="🩺", layout="wide")
st.title("SCAN A — Intelligent Ultrasound Digital Twin")
st.caption("Acquisition → Digital Signature → Statistical Baseline → Anomaly Detection → Digital Twin State → Drift → Trend → Validation")

def load_json(name: str) -> dict:
    path = OUTPUT_DIR / name
    if not path.exists(): return {}
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return {}

def load_csv(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists(): return pd.DataFrame()
    try: return pd.read_csv(path)
    except (OSError, pd.errors.ParserError): return pd.DataFrame()

summary = load_json("summary.json")
states = load_csv("twin_states.csv")
features = load_csv("features.csv")

with st.sidebar:
    st.header("Digital Twin")
    st.metric("Data source", str(summary.get("source", "unknown")).upper())
    st.metric("Acquisitions", summary.get("n_acquisitions", "—"))
    st.divider()
    st.markdown("**Interpretation**")
    st.caption("A deviation is a statistical/engineering observation. It is not, by itself, proof of a hardware fault or a clinical diagnosis.")

if not summary and states.empty:
    st.warning("No pipeline outputs found. Run python main.py first, then refresh.")
    st.stop()

latest_state, latest_distance, latest_quality, drift_status = "—", None, None, "—"
if not states.empty:
    last = states.iloc[-1]
    latest_state = str(last.get("state", "—"))
    latest_distance = last.get("mahalanobis_distance")
    latest_quality = last.get("quality_score")
    drift_status = str(last.get("drift_status", "—"))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Digital Twin state", latest_state)
c2.metric("Mahalanobis distance", "—" if pd.isna(latest_distance) else f"{float(latest_distance):.3f}")
c3.metric("Quality / conformity", "—" if pd.isna(latest_quality) else f"{float(latest_quality):.1f}")
c4.metric("Drift status", drift_status)

tab_monitor, tab_signature, tab_report = st.tabs(["📈 Monitoring", "🧬 Digital Signature", "📄 Report"])

with tab_monitor:
    st.subheader("Temporal monitoring")
    if states.empty:
        st.info("No twin_states.csv artifact available.")
    else:
        numeric_cols = [c for c in ["mahalanobis_distance", "mahalanobis_squared", "quality_score"] if c in states.columns]
        if numeric_cols:
            chart = states[numeric_cols].copy()
            chart.index = states["acquisition_id"].astype(str) if "acquisition_id" in states.columns else chart.index
            st.line_chart(chart, height=360)
        display_cols = [c for c in ["acquisition_id","source","state","mahalanobis_distance","mahalanobis_squared","quality_score","drift_status"] if c in states.columns]
        st.dataframe(states[display_cols], use_container_width=True, hide_index=True)

with tab_signature:
    st.subheader("Digital Signature")
    if features.empty:
        st.info("No features.csv artifact available. The pipeline may have been run without feature export.")
    else:
        st.dataframe(features, use_container_width=True, hide_index=True)
        numeric = features.select_dtypes(include="number")
        if not numeric.empty:
            st.subheader("Latest feature profile")
            st.bar_chart(numeric.iloc[-1])

with tab_report:
    report_path = OUTPUT_DIR / "report.md"
    if report_path.exists(): st.markdown(report_path.read_text(encoding="utf-8"))
    else: st.info("No Markdown report found.")

st.divider()
st.caption("SCAN A Digital Twin V3 presentation layer — engineering research and equipment monitoring.")
