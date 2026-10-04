"""SCAN A Digital Twin — Expert Engineering Dashboard V4."""
from __future__ import annotations
from pathlib import Path
import json
import math
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

st.set_page_config(page_title="SCAN A | Digital Twin", page_icon="◈", layout="wide")

st.markdown("""
<style>
.stApp{background:#07111f;color:#edf4fb}
[data-testid="stHeader"]{background:#07111f}
.block-container{max-width:1500px;padding:1.2rem 2rem 3rem}
section[data-testid="stSidebar"]{background:#091625;border-right:1px solid #20334a}
.hero,.panel,.kpi,.status{background:#0c1929;border:1px solid #20334a;border-radius:14px}
.hero{padding:25px 28px;margin-bottom:18px;background:linear-gradient(135deg,#10233a,#0b1727)}
.hero h1{margin:5px 0;font-size:2rem;letter-spacing:-.02em}
.hero p,.muted,.small{color:#8fa4ba}
.eyebrow,.section{color:#42c7b7;font-size:.7rem;font-weight:800;letter-spacing:.15em;text-transform:uppercase}
.section{margin:24px 0 9px 2px}
.kpi{padding:15px 17px;min-height:105px}
.kpi-label{color:#8fa4ba;font-size:.68rem;text-transform:uppercase;letter-spacing:.1em}
.kpi-value{font-size:1.5rem;font-weight:800;margin-top:7px}
.kpi-note{color:#8fa4ba;font-size:.72rem;margin-top:4px}
.status{padding:18px 20px;background:#0b2230}
.state{font-size:1.6rem;font-weight:850;margin-top:4px}
.badge{display:inline-block;padding:4px 9px;border-radius:999px;font-size:.65rem;font-weight:800;letter-spacing:.08em}
.green{color:#68ddca;background:#0c2928;border:1px solid #24675f}
.yellow{color:#f0c56b;background:#2b2310;border:1px solid #665025}
.red{color:#ff858c;background:#2d1519;border:1px solid #6d3036}
.pipeline{display:flex;gap:7px;align-items:center;flex-wrap:wrap}
.node{padding:7px 10px;border-radius:8px;background:#101f33;border:1px solid #20334a;font-size:.72rem}
.arrow{color:#50657b}
.stTabs [data-baseweb="tab-list"]{gap:5px}
.stTabs [data-baseweb="tab"]{color:#8fa4ba}
</style>
""", unsafe_allow_html=True)

def load_json(name):
    p = OUT / name
    if not p.exists(): return {}
    try: return json.loads(p.read_text(encoding="utf-8"))
    except Exception: return {}

def load_csv(name):
    p = OUT / name
    if not p.exists(): return pd.DataFrame()
    try: return pd.read_csv(p)
    except Exception: return pd.DataFrame()

summary = load_json("summary.json")
states = load_csv("twin_states.csv")
features = load_csv("features.csv")
report = OUT / "report.md"

if not summary and states.empty:
    st.error("No pipeline artifacts. Run: python main.py")
    st.stop()

latest = states.iloc[-1] if not states.empty else pd.Series(dtype=object)

def val(key, fallback="—"):
    x = latest.get(key, summary.get(key, fallback))
    if x is None: return fallback
    try:
        if pd.isna(x): return fallback
    except Exception: pass
    return x

def number(key, digits=3):
    try: return f"{float(val(key)):.{digits}f}"
    except Exception: return "—"

source = str(val("source", "unknown")).upper()
state = str(val("state"))
acq_id = str(val("acquisition_id"))
drift = str(val("drift_status"))
state_class = "green" if state.upper() in ("NOMINAL","NORMAL") else ("red" if "ANOM" in state.upper() or "ALERT" in state.upper() else "yellow")

st.markdown(
    '<div class="hero"><div class="eyebrow">ENGINEERING MONITORING CONSOLE · V4</div>'
    '<h1>SCAN A — Intelligent Ultrasound Digital Twin</h1>'
    '<p>Condition assessment · Digital Signature · Multivariate monitoring · Drift surveillance · Validation</p></div>',
    unsafe_allow_html=True
)

if source in ("SIMULATED","DEMO"):
    st.info("SIMULATION MODE — Current observations validate the software workflow. They are not measurements proving a physical SCAN A fault.")

st.markdown('<div class="section">System configuration</div>', unsafe_allow_html=True)
c = st.columns(5)
items = [
    ("Operating state", state, "Current Digital Twin classification"),
    ("Acquisitions", val("n_acquisitions", len(states)), "Total observations"),
    ("Baseline", val("baseline_size", val("baseline")), "Reference set"),
    ("Calibration", val("calibration_size", val("calibration")), "Calibration set"),
    ("Test set", val("test_size", val("test")), "Assessed observations"),
]
for col, item in zip(c, items):
    col.markdown('<div class="kpi"><div class="kpi-label">'+str(item[0])+
                 '</div><div class="kpi-value">'+str(item[1])+
                 '</div><div class="kpi-note">'+str(item[2])+'</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section">Current condition assessment</div>', unsafe_allow_html=True)
a,b,d,e = st.columns(4)
a.markdown('<div class="status"><div class="eyebrow">DIGITAL TWIN STATE</div><div class="state">'+state+
           '</div><div class="small">Acquisition '+acq_id+' · source '+source+
           '</div><br><span class="badge '+state_class+'">'+state.upper()+'</span></div>', unsafe_allow_html=True)
b.markdown('<div class="kpi"><div class="kpi-label">Mahalanobis distance</div><div class="kpi-value">'+number("mahalanobis_distance",4)+
           '</div><div class="kpi-note">Multivariate separation from the learned feature distribution.</div></div>', unsafe_allow_html=True)
d.markdown('<div class="kpi"><div class="kpi-label">Mahalanobis D²</div><div class="kpi-value">'+number("mahalanobis_squared",4)+
           '</div><div class="kpi-note">Squared statistical distance used by the monitoring layer.</div></div>', unsafe_allow_html=True)
e.markdown('<div class="kpi"><div class="kpi-label">Quality / conformity</div><div class="kpi-value">'+number("quality_score",2)+
           '</div><div class="kpi-note">Pipeline score; not a hardware-health percentage.</div></div>', unsafe_allow_html=True)

tabs = st.tabs(["Overview","Acquisition Explorer","Digital Signature","Statistical Monitoring","Drift & Trend","Validation","Engineering Report"])

with tabs[0]:
    st.markdown('<div class="section">Digital Twin architecture</div>', unsafe_allow_html=True)
    nodes = ["Acquisition","Digital Signature","Statistical Baseline","Anomaly Detection","Twin State","Drift","Trend"]
    html = '<div class="panel"><div class="pipeline">'
    for i,n in enumerate(nodes):
        html += '<span class="node">'+n+'</span>'
        if i < len(nodes)-1: html += '<span class="arrow">→</span>'
    html += '</div></div>'
    st.markdown(html, unsafe_allow_html=True)
    x,y = st.columns([1.4,1])
    x.markdown('<div class="panel"><b>Engineering interpretation</b><p class="muted">'
               'The displayed state is the output of the existing statistical pipeline. '
               'Its meaning is bounded by the current validation level and data provenance.</p>'
               '<b>Latest acquisition:</b> '+acq_id+'<br><b>Source:</b> '+source+
               '<br><b>Drift:</b> '+drift+'</div>', unsafe_allow_html=True)
    y.markdown('<div class="panel"><b>Scientific boundary</b><p class="muted">'
               'Simulation and public datasets validate software behavior. They do not establish '
               'the physical SCAN A baseline, hardware thresholds, or a clinical diagnosis.</p></div>', unsafe_allow_html=True)

with tabs[1]:
    st.markdown('<div class="section">Acquisition explorer</div>', unsafe_allow_html=True)
    if states.empty:
        st.warning("No twin_states.csv artifact.")
    else:
        ids = states["acquisition_id"].astype(str).tolist() if "acquisition_id" in states else [str(i) for i in states.index]
        selected = st.selectbox("Select acquisition", ids, index=len(ids)-1)
        row = states.iloc[ids.index(selected)]
        cols = st.columns(4)
        specs = [("State","state",""),("Mahalanobis D","mahalanobis_distance",".4f"),
                 ("Mahalanobis D²","mahalanobis_squared",".4f"),("Quality score","quality_score",".2f")]
        for col,(label,key,fmt) in zip(cols,specs):
            raw = row.get(key,"—")
            try: shown = format(float(raw),fmt) if fmt else str(raw)
            except Exception: shown = str(raw)
            col.markdown('<div class="kpi"><div class="kpi-label">'+label+
                         '</div><div class="kpi-value">'+shown+'</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="section">Acquisition record</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([row]), use_container_width=True, hide_index=True)

with tabs[2]:
    st.markdown('<div class="section">Digital Signature · feature space</div>', unsafe_allow_html=True)
    if features.empty:
        st.warning("No features.csv artifact.")
    else:
        ids = features["acquisition_id"].astype(str).tolist() if "acquisition_id" in features else [str(i) for i in features.index]
        selected = st.selectbox("Select signature", ids, index=len(ids)-1, key="signature")
        row = features.iloc[ids.index(selected)]
        numeric = features.select_dtypes(include="number")
        names = [n for n in numeric.columns if n != "acquisition_index"]
        cols = st.columns(3)
        for i,name in enumerate(names):
            raw = row.get(name,"—")
            try: shown = f"{float(raw):.5g}"
            except Exception: shown = str(raw)
            cols[i%3].markdown('<div class="kpi"><div class="kpi-label">'+name.replace("_"," ")+
                                '</div><div class="kpi-value">'+shown+'</div></div>', unsafe_allow_html=True)
        if names:
            st.markdown('<div class="section">Feature profile</div>', unsafe_allow_html=True)
            profile = pd.DataFrame({"value":[row[n] for n in names]}, index=names)
            st.bar_chart(profile, height=380)
        st.markdown('<div class="section">Complete signature matrix</div>', unsafe_allow_html=True)
        st.dataframe(features, use_container_width=True, hide_index=True)

with tabs[3]:
    st.markdown('<div class="section">Multivariate statistical monitoring</div>', unsafe_allow_html=True)
    if states.empty:
        st.warning("No state history available.")
    else:
        numeric = [x for x in ["mahalanobis_distance","mahalanobis_squared","quality_score"] if x in states.columns]
        if numeric:
            chart = states[numeric].copy()
            if "acquisition_id" in states: chart.index = states["acquisition_id"].astype(str)
            st.line_chart(chart, height=430)
        display = [x for x in ["acquisition_id","source","state","mahalanobis_distance","mahalanobis_squared","quality_score","drift_status"] if x in states.columns]
        st.dataframe(states[display], use_container_width=True, hide_index=True)
        st.markdown('<div class="panel"><b>Interpretation</b><p class="muted">'
                    'Mahalanobis distance measures multivariate separation from the learned feature distribution. '
                    'Higher distance means greater statistical separation; engineering significance depends on baseline, calibration and validation.</p></div>',
                    unsafe_allow_html=True)

with tabs[4]:
    st.markdown('<div class="section">Drift & temporal surveillance</div>', unsafe_allow_html=True)
    if states.empty:
        st.warning("No temporal history available.")
    else:
        if "drift_status" in states.columns:
            counts = states["drift_status"].astype(str).value_counts().rename_axis("status").to_frame("acquisitions")
            st.dataframe(counts, use_container_width=True)
        if "quality_score" in states.columns:
            q = states[["quality_score"]].copy()
            if "acquisition_id" in states: q.index = states["acquisition_id"].astype(str)
            st.line_chart(q, height=320)
        st.markdown('<div class="panel"><b>Engineering meaning</b><p class="muted">'
                    'Temporal monitoring follows changes in observed statistical behavior. '
                    'It is not a failure-date predictor and does not establish RUL or physical failure.</p></div>',
                    unsafe_allow_html=True)

with tabs[5]:
    st.markdown('<div class="section">Validation maturity</div>', unsafe_allow_html=True)
    checks = [
        ("Software pipeline","COMPLETE","Execution and artifact generation."),
        ("Simulation","COMPLETE","Controlled synthetic workflow validation."),
        ("Public ultrasound benchmark","COMPLETE","Methodological/software validation only."),
        ("Real SCAN A acquisition","PENDING","Requires repeated controlled acquisitions from the physical system."),
        ("Controlled perturbation","PENDING","Only if safe, authorized and experimentally documented."),
    ]
    for name,status,note in checks:
        cls = "green" if status == "COMPLETE" else "yellow"
        st.markdown('<div class="panel" style="margin-bottom:8px"><span class="badge '+cls+'">'+status+
                    '</span> <b>'+name+'</b><br><span class="small">'+note+'</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="section">Data provenance</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel"><b>Current source:</b> '+source+
                '<br><span class="small">Synthetic/public data must not silently define the physical SCAN A Digital Signature or hardware thresholds.</span></div>',
                unsafe_allow_html=True)

with tabs[6]:
    st.markdown('<div class="section">Engineering report</div>', unsafe_allow_html=True)
    if report.exists():
        st.markdown(report.read_text(encoding="utf-8"))
    else:
        st.info("No report.md artifact.")

st.divider()
st.caption("SCAN A Digital Twin V4 · Presentation layer only · Engineering monitoring, not clinical diagnosis or automatic proof of hardware failure.")
