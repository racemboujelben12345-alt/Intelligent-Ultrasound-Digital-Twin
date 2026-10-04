"""SCAN A Digital Twin — Engineering Monitoring Workstation.

Presentation layer only. The scientific pipeline and artifacts remain unchanged.
"""
from __future__ import annotations

from pathlib import Path
import json
import math
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

st.set_page_config(
    page_title="SCAN A | Engineering Workstation",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root{
  --bg:#f4f7fa; --surface:#ffffff; --surface2:#eef4f8; --line:#d7e1e8;
  --text:#172433; --muted:#66798a; --blue:#1769AA; --teal:#087f83;
  --warn:#b77b12; --danger:#bd3f4b; --info:#2d6fb7; --shadow:0 2px 10px rgba(23,36,51,.06);
}
.stApp{background:var(--bg);color:var(--text)}
[data-testid="stHeader"]{background:rgba(244,247,250,.96)}
.block-container{max-width:1600px;padding:1.2rem 2rem 3rem}
section[data-testid="stSidebar"]{background:#ffffff;border-right:1px solid var(--line)}
section[data-testid="stSidebar"] .block-container{padding-top:1.2rem}
h1,h2,h3,h4{color:var(--text)}
.small,.muted{color:var(--muted)}
.eyebrow{color:var(--blue);font-size:.65rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase}
.hero{background:#fff;border:1px solid var(--line);border-left:5px solid var(--blue);
      border-radius:12px;padding:22px 26px;margin-bottom:12px;box-shadow:var(--shadow)}
.hero h1{margin:.25rem 0 .35rem;font-size:2rem;letter-spacing:-.025em}
.hero p{margin:0;color:var(--muted);font-size:.9rem}
.banner{border:1px solid #e5cf9a;background:#fff9ea;border-radius:9px;padding:10px 13px;color:#77570d}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px;box-shadow:var(--shadow)}
.metric-label{font-size:.62rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em}
.metric-value{font-size:1.5rem;font-weight:800;line-height:1.2;margin-top:7px;color:var(--text)}
.metric-note{font-size:.67rem;color:var(--muted);margin-top:5px}
.section{margin:21px 0 9px 1px;color:#496579;font-size:.66rem;font-weight:800;
         letter-spacing:.14em;text-transform:uppercase}
.state-card{background:#f8fbfd;border:1px solid #bfd4e1;border-left:4px solid var(--teal);
            border-radius:10px;padding:17px 19px;box-shadow:var(--shadow)}
.state{font-size:1.75rem;font-weight:900;letter-spacing:.01em;margin:.15rem 0}
.badge{display:inline-block;border-radius:999px;padding:4px 9px;font-size:.6rem;font-weight:850;letter-spacing:.08em}
.ok{color:#176b55;background:#e8f6f0;border:1px solid #abd9c8}
.watch{color:#8a5d09;background:#fff3d6;border:1px solid #ead08c}
.alert{color:#9c303b;background:#fdebed;border:1px solid #e5b1b8}
.neutral{color:#526b7c;background:#edf2f5;border:1px solid #ccd8df}
.pipeline{display:flex;align-items:center;gap:7px;flex-wrap:wrap;background:#fff;
          border:1px solid var(--line);border-radius:10px;padding:13px;box-shadow:var(--shadow)}
.node{padding:7px 10px;background:#f1f6f9;border:1px solid #ccdce5;border-radius:6px;
      font-size:.68rem;color:#29465b;font-weight:650}
.arrow{color:#8aa0af}
.trace{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.72rem;color:#526b7c}
div[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);padding:9px 12px;border-radius:9px;box-shadow:var(--shadow)}
.stTabs [data-baseweb="tab-list"]{gap:3px;background:#e9f0f4;padding:4px;border-radius:8px}
.stTabs [data-baseweb="tab"]{color:#607687;font-size:.74rem}
.stTabs [aria-selected="true"]{color:var(--blue);font-weight:750}
hr{border-color:var(--line)}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:8px}
</style>
""",
    unsafe_allow_html=True,
)

def load_json(name: str) -> dict:
    p = OUT / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def load_csv(name: str) -> pd.DataFrame:
    p = OUT / name
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

summary = load_json("summary.json")
states = load_csv("twin_states.csv")
features = load_csv("features.csv")
report_path = OUT / "report.md"

if states.empty and not summary:
    st.error("No pipeline artifacts found. Run: python main.py")
    st.stop()

latest = states.iloc[-1] if not states.empty else pd.Series(dtype=object)

def get_value(key: str, fallback="—"):
    x = latest.get(key, summary.get(key, fallback))
    try:
        if pd.isna(x):
            return fallback
    except Exception:
        pass
    return x

def fmt(key: str, digits: int = 3) -> str:
    try:
        return f"{float(get_value(key)):.{digits}f}"
    except Exception:
        return "—"

def safe_int(value, fallback=0):
    try:
        return int(float(value))
    except Exception:
        return fallback

source = str(get_value("source", "unknown")).upper()
state = str(get_value("state", "UNKNOWN")).upper()
acq_id = str(get_value("acquisition_id", "—"))
drift = str(get_value("drift_status", "—")).upper()
n_acq = safe_int(get_value("n_acquisitions", len(states)), len(states))

baseline_size = safe_int(summary.get("baseline_size", summary.get("baseline", get_value("baseline_size", 0))), 0)
calibration_size = safe_int(summary.get("calibration_size", summary.get("calibration", get_value("calibration_size", 0))), 0)
test_size = safe_int(summary.get("test_size", summary.get("test", get_value("test_size", 0))), 0)

if state in {"NOMINAL", "NORMAL"}:
    state_cls = "ok"
elif any(x in state for x in ("ANOM", "ALERT", "FAIL")):
    state_cls = "alert"
else:
    state_cls = "watch"

source_label = "SIMULATION" if source in {"SIMULATED", "DEMO"} else source

with st.sidebar:
    st.markdown('<div class="eyebrow">SCAN A / DIGITAL TWIN</div>', unsafe_allow_html=True)
    st.markdown("### Engineering Workstation")
    st.markdown('<div class="small">Condition assessment and multivariate surveillance</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("**DATA PROVENANCE**")
    st.markdown(
        f'<span class="badge {"watch" if source_label == "SIMULATION" else "ok"}">{source_label}</span>',
        unsafe_allow_html=True,
    )
    st.caption("Current artifacts")
    st.markdown("**VALIDATION MATURITY**")
    st.markdown("Software pipeline · **COMPLETE**")
    st.markdown("Simulation · **COMPLETE**")
    st.markdown("Public benchmark · **COMPLETE**")
    st.markdown("Real SCAN A · **PENDING**")
    st.markdown("Physical baseline · **PENDING**")
    st.divider()
    st.markdown("**PIPELINE**")
    st.caption("Acquisition → Signature → Baseline → Detection → Twin State → Drift → Trend")
    st.caption("Presentation layer does not modify the scientific engine.")

st.markdown(
    '<div class="hero">'
    '<div class="eyebrow">ENGINEERING MONITORING WORKSTATION · V5.1</div>'
    '<h1>SCAN A — Intelligent Ultrasound Digital Twin</h1>'
    '<p>Condition assessment · Digital Signature · Statistical monitoring · Drift surveillance · Validation maturity</p>'
    '</div>',
    unsafe_allow_html=True,
)

if source_label == "SIMULATION":
    st.markdown(
        '<div class="banner"><b>SIMULATION MODE</b> · Synthetic observations validate the '
        'software workflow only. They do not establish a physical SCAN A fault, threshold, '
        'or hardware-health percentage.</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="section">System state</div>', unsafe_allow_html=True)
cols = st.columns([1.45, 1, 1, 1, 1, 1])
cols[0].markdown(
    f'<div class="state-card"><div class="eyebrow">DIGITAL TWIN STATE</div>'
    f'<div class="state">{state}</div><span class="badge {state_cls}">{state}</span>'
    f'<div class="small" style="margin-top:8px">Latest acquisition · {acq_id}</div></div>',
    unsafe_allow_html=True,
)
metrics = [
    ("Acquisitions", n_acq, "observations"),
    ("Baseline", baseline_size, "reference observations"),
    ("Calibration", calibration_size, "calibration observations"),
    ("Test", test_size, "assessed observations"),
    ("Source", source_label, "data provenance"),
]
for col, (label, value, note) in zip(cols[1:], metrics):
    col.markdown(
        f'<div class="card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="section">Current condition assessment</div>', unsafe_allow_html=True)
a, b, c, d = st.columns(4)
a.metric("Mahalanobis D", fmt("mahalanobis_distance", 4), help="Multivariate statistical distance from the learned feature distribution.")
b.metric("Mahalanobis D²", fmt("mahalanobis_squared", 4), help="Squared Mahalanobis distance.")
c.metric("Quality / conformity", fmt("quality_score", 2), help="Pipeline score; not a device-health percentage.")
d.metric("Drift status", drift, help="Current drift classification produced by the existing pipeline.")

st.markdown('<div class="section">Digital twin chain</div>', unsafe_allow_html=True)
nodes = ["Acquisition", "Digital Signature", "Statistical Baseline", "Anomaly Detection",
         "Twin State", "Drift", "Trend / Prediction"]
html = '<div class="pipeline">'
for i, node in enumerate(nodes):
    html += f'<span class="node">{node}</span>'
    if i < len(nodes) - 1:
        html += '<span class="arrow">→</span>'
html += "</div>"
st.markdown(html, unsafe_allow_html=True)

tabs = st.tabs([
    "Command Center", "Acquisition Trace", "Digital Signature",
    "Multivariate Monitoring", "Drift Surveillance", "Validation",
    "Engineering Report",
])

with tabs[0]:
    st.markdown('<div class="section">Operational overview</div>', unsafe_allow_html=True)
    left, right = st.columns([1.55, 1])
    with left:
        if not states.empty:
            chart_cols = [x for x in ["mahalanobis_distance", "quality_score"] if x in states.columns]
            if chart_cols:
                chart = states[chart_cols].copy()
                if "acquisition_id" in states:
                    chart.index = states["acquisition_id"].astype(str)
                st.line_chart(chart, height=330)
        st.markdown(
            '<div class="small">Observed trajectory from generated twin states. '
            'No physical acceptance threshold is displayed unless established from real SCAN A validation.</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<div class="card"><div class="eyebrow">ENGINEERING INTERPRETATION</div>'
            f'<h3 style="margin:.35rem 0">Current state: {state}</h3>'
            f'<p class="muted">Acquisition <b>{acq_id}</b> is currently classified by the existing '
            f'multivariate pipeline as <b>{state}</b>. The statistical distance and drift outputs '
            'must be interpreted against a validated SCAN A baseline.</p>'
            f'<div class="trace">SOURCE  {source_label}<br>DRIFT   {drift}<br>'
            f'FEATURES {len([x for x in features.columns if x not in {"acquisition_id","acquisition_index"}]) if not features.empty else "—"}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="card"><div class="eyebrow">SCIENTIFIC BOUNDARY</div>'
            '<p class="muted">Public and synthetic ultrasound data support methodological and software '
            'validation. They must not silently define the physical SCAN A Digital Signature, hardware '
            'thresholds, or a diagnosis.</p></div>',
            unsafe_allow_html=True,
        )

with tabs[1]:
    st.markdown('<div class="section">Acquisition traceability</div>', unsafe_allow_html=True)
    if states.empty:
        st.warning("No twin_states.csv artifact.")
    else:
        ids = states["acquisition_id"].astype(str).tolist() if "acquisition_id" in states else [str(i) for i in states.index]
        selected = st.selectbox("Acquisition record", ids, index=len(ids)-1)
        row = states.iloc[ids.index(selected)]
        c1, c2, c3, c4 = st.columns(4)
        for col, label, key, digits in [
            (c1, "State", "state", None), (c2, "Mahalanobis D", "mahalanobis_distance", 4),
            (c3, "Mahalanobis D²", "mahalanobis_squared", 4), (c4, "Quality / conformity", "quality_score", 2),
        ]:
            raw = row.get(key, "—")
            try:
                shown = f"{float(raw):.{digits}f}" if digits is not None else str(raw)
            except Exception:
                shown = str(raw)
            col.metric(label, shown)
        st.markdown('<div class="section">Record payload</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([row]), use_container_width=True, hide_index=True)
        st.markdown('<div class="section">Acquisition position</div>', unsafe_allow_html=True)
        try:
            idx = ids.index(selected)
            stage = "BASELINE" if idx < baseline_size else "CALIBRATION" if idx < baseline_size + calibration_size else "TEST"
            st.markdown(
                f'<div class="card"><span class="badge neutral">{stage}</span> '
                f'<span class="small">Sequence position {idx + 1} / {len(ids)}</span></div>',
                unsafe_allow_html=True,
            )
        except Exception:
            pass

with tabs[2]:
    st.markdown('<div class="section">Digital Signature · feature-space inspection</div>', unsafe_allow_html=True)
    if features.empty:
        st.warning("No features.csv artifact.")
    else:
        ids = features["acquisition_id"].astype(str).tolist() if "acquisition_id" in features else [str(i) for i in features.index]
        selected = st.selectbox("Signature acquisition", ids, index=len(ids)-1, key="signature_v5")
        row = features.iloc[ids.index(selected)]
        excluded = {"acquisition_id", "acquisition_index", "source"}
        names = [n for n in features.columns if n not in excluded and pd.api.types.is_numeric_dtype(features[n])]
        bdf = features.iloc[:baseline_size] if baseline_size > 1 else features.iloc[:0]
        baseline_stats = {}
        if not bdf.empty:
            for n in names:
                mu = pd.to_numeric(bdf[n], errors="coerce").mean()
                sd = pd.to_numeric(bdf[n], errors="coerce").std(ddof=1)
                baseline_stats[n] = (mu, sd)
        if names:
            top = st.columns(4)
            for i, name in enumerate(names[:12]):
                raw = row.get(name, "—")
                try:
                    shown = f"{float(raw):.5g}"
                except Exception:
                    shown = str(raw)
                top[i % 4].metric(name.replace("_", " "), shown)
            st.markdown('<div class="section">Signature profile</div>', unsafe_allow_html=True)
            profile = pd.DataFrame({"value": [pd.to_numeric(row[n], errors="coerce") for n in names]}, index=names)
            st.bar_chart(profile, height=360)
            if baseline_stats:
                records = []
                for n in names:
                    mu, sd = baseline_stats[n]
                    value = pd.to_numeric(row[n], errors="coerce")
                    z = (value - mu) / sd if sd and not math.isnan(sd) else float("nan")
                    records.append({"feature": n, "value": value, "baseline_mean": mu,
                                    "baseline_std": sd, "deviation_sigma": z})
                dev = pd.DataFrame(records).sort_values("deviation_sigma", key=lambda s: s.abs(), ascending=False)
                st.markdown('<div class="section">Deviation from current software baseline partition</div>', unsafe_allow_html=True)
                st.dataframe(dev, use_container_width=True, hide_index=True)
        st.markdown('<div class="section">Complete signature matrix</div>', unsafe_allow_html=True)
        st.dataframe(features, use_container_width=True, hide_index=True)

with tabs[3]:
    st.markdown('<div class="section">Multivariate monitoring</div>', unsafe_allow_html=True)
    if states.empty:
        st.warning("No state history available.")
    else:
        c1, c2 = st.columns([1.6, 1])
        with c1:
            numeric = [x for x in ["mahalanobis_distance", "mahalanobis_squared", "quality_score"] if x in states.columns]
            if numeric:
                chart = states[numeric].copy()
                if "acquisition_id" in states:
                    chart.index = states["acquisition_id"].astype(str)
                st.line_chart(chart, height=400)
        with c2:
            st.markdown(
                '<div class="card"><div class="eyebrow">STATISTICAL MEANING</div>'
                '<p class="muted">Mahalanobis distance summarizes multivariate separation of an observation '
                'from the learned feature distribution. It is a statistical monitoring quantity, not a '
                'direct physical measurement.</p><div class="trace">D² = D × D</div></div>',
                unsafe_allow_html=True,
            )
            if "state" in states.columns:
                counts = states["state"].astype(str).value_counts().rename_axis("state").to_frame("count")
                st.dataframe(counts, use_container_width=True)
        display = [x for x in [
            "acquisition_id", "source", "state", "mahalanobis_distance",
            "mahalanobis_squared", "quality_score", "drift_status"
        ] if x in states.columns]
        st.dataframe(states[display], use_container_width=True, hide_index=True)

with tabs[4]:
    st.markdown('<div class="section">Drift surveillance & temporal behavior</div>', unsafe_allow_html=True)
    if states.empty:
        st.warning("No temporal history available.")
    else:
        if "drift_status" in states.columns:
            drift_counts = states["drift_status"].astype(str).value_counts().rename_axis("drift_status").to_frame("observations")
            st.dataframe(drift_counts, use_container_width=True, hide_index=True)
        if "quality_score" in states.columns:
            q = states[["quality_score"]].copy()
            if "acquisition_id" in states:
                q.index = states["acquisition_id"].astype(str)
            st.markdown('<div class="section">Quality trajectory</div>', unsafe_allow_html=True)
            st.line_chart(q, height=300)
        st.markdown(
            '<div class="card"><div class="eyebrow">DRIFT INTERPRETATION</div>'
            '<p class="muted">Temporal surveillance identifies changes in the statistical behavior of '
            'the observed feature signature. Persistence and direction are useful for engineering review, '
            'but the current synthetic workflow cannot be interpreted as proof of a physical SCAN A drift.</p>'
            '<p class="muted"><b>Not RUL · not a failure-date predictor · not a hardware diagnosis.</b></p></div>',
            unsafe_allow_html=True,
        )

with tabs[5]:
    st.markdown('<div class="section">Validation maturity matrix</div>', unsafe_allow_html=True)
    checks = [
        ("Software pipeline", "COMPLETE", "Execution and artifact generation."),
        ("Synthetic simulation", "COMPLETE", "Controlled workflow and statistical-engine behavior."),
        ("Public ultrasound benchmark", "COMPLETE", "Methodological/software validation only."),
        ("Real SCAN A acquisition", "PENDING", "Repeated controlled acquisitions from the physical system."),
        ("SCAN A Digital Signature", "PENDING", "Requires real-system characterization and repeatability."),
        ("Physical baseline / thresholds", "PENDING", "Must be established from validated real SCAN A observations."),
        ("Controlled perturbation study", "PENDING", "Only when safe, authorized and experimentally documented."),
    ]
    for name, status, note in checks:
        cls = "ok" if status == "COMPLETE" else "watch"
        st.markdown(
            f'<div class="card" style="margin-bottom:8px"><span class="badge {cls}">{status}</span> '
            f'<b>{name}</b><div class="small" style="margin-top:5px">{note}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="section">Data governance</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="card"><div class="trace">CURRENT SOURCE  {source_label}</div>'
        '<p class="muted">Synthetic and public datasets are explicitly separated from the future '
        'physical SCAN A baseline. This prevents benchmark behavior from being presented as machine-specific evidence.</p></div>',
        unsafe_allow_html=True,
    )

with tabs[6]:
    st.markdown('<div class="section">Generated engineering report</div>', unsafe_allow_html=True)
    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))
    else:
        st.info("No report.md artifact available.")

st.divider()
st.caption(
    "SCAN A Digital Twin V5.1 · Presentation layer only · Engineering monitoring workstation · "
    "Not clinical diagnosis, not automatic proof of hardware failure, not RUL."
)
