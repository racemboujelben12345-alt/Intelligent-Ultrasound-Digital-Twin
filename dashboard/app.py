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

st.set_page_config(page_title="SCAN A | Engineering Workstation", page_icon="◈", layout="wide")

# --- Engineering workstation visual layer ---
st.markdown(
    """
    <style>
    .stApp { background:#f3f7fa; }
    [data-testid="stHeader"] { background:transparent; }
    .block-container { padding-top:.65rem; max-width:1500px; }

    /* Single clean navy header: intentionally no grey band above it. */
    .scan-header {
        margin:-.65rem -3rem 1.15rem -3rem;
        min-height:78px;
        padding:12px 34px;
        display:flex;
        align-items:center;
        justify-content:space-between;
        background:linear-gradient(100deg,#092844 0%,#0d3558 58%,#123f68 100%);
        border-bottom:4px solid #1d73c9;
        box-shadow:0 5px 18px rgba(10,42,70,.12);
        color:white;
    }
    .scan-brand { display:flex; align-items:center; gap:15px; }
    .scan-logo {
        width:45px;height:45px;border-radius:10px;
        display:grid;place-items:center;
        border:1px solid rgba(255,255,255,.28);
        background:rgba(255,255,255,.08);
        font-size:25px;font-weight:700;
    }
    .scan-name { font-size:20px;font-weight:800;letter-spacing:.05em;line-height:1; }
    .scan-sub {
        margin-top:6px;font-size:10px;letter-spacing:.12em;
        text-transform:uppercase;opacity:.76;
    }
    .scan-status {
        padding:9px 14px;border-radius:999px;
        border:1px solid rgba(255,255,255,.27);
        background:rgba(255,255,255,.07);
        font-size:11px;font-weight:800;letter-spacing:.06em;
    }
    .scan-dot {
        display:inline-block;width:8px;height:8px;border-radius:50%;
        background:#31c58a;margin-right:7px;
    }
    .scan-hero {
        background:#fff;border:1px solid #d7e2eb;border-left:5px solid #1d73c9;
        border-radius:7px;padding:20px 24px;
        box-shadow:0 2px 10px rgba(24,56,80,.045);
    }
    .scan-kicker {
        color:#1d73c9;font-size:10px;font-weight:800;
        letter-spacing:.17em;text-transform:uppercase;
    }
    .scan-hero h1 { margin:6px 0 5px; color:#172b3d; font-size:29px; }
    .scan-hero p { margin:0;color:#6b7d8d;font-size:13px; }
    .scan-section {
        margin:17px 0 9px;color:#31516b;font-size:10px;font-weight:800;
        letter-spacing:.16em;text-transform:uppercase;
        border-bottom:1px solid #d7e2eb;padding-bottom:7px;
    }
    .scan-notice {
        margin:12px 0 18px;padding:11px 14px;
        background:#fff8e8;border:1px solid #efd69b;border-radius:6px;
        color:#6c5212;font-size:12px;line-height:1.45;
    }
    [data-testid="stSidebar"] { background:#f8fafc;border-right:1px solid #d7e2eb; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="scan-header">
      <div class="scan-brand">
        <div class="scan-logo">⌁</div>
        <div>
          <div class="scan-name">SCAN A</div>
          <div class="scan-sub">Intelligent Ultrasound Digital Twin · Biomedical Engineering</div>
        </div>
      </div>
      <div class="scan-status"><span class="scan-dot"></span>SIMULATION · NOMINAL</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="scan-hero">
      <div class="scan-kicker">SYSTEM OVERVIEW</div>
      <h1>Engineering condition assessment</h1>
      <p>Digital Signature · Multivariate Monitoring · Drift Surveillance · Validation Maturity</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="scan-notice">
      <b>SIMULATION MODE:</b>
      Synthetic observations validate the software workflow only.
      They do not establish a physical SCAN A fault, threshold, or hardware-health percentage.
    </div>
    """,
    unsafe_allow_html=True,
)


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
st.caption("SCAN A Digital Twin · Engineering research workstation · Simulation evidence is explicitly separated from future real SCAN A validation.")
