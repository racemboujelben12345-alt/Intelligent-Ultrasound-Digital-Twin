"""SCAN A Digital Twin — Engineering Monitoring Console V5.

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
    page_title="SCAN A | Engineering Console",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Visual system — restrained instrumentation-console style
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
:root{
  --bg:#07101b; --surface:#0b1725; --surface2:#0f1d2d; --line:#20354a;
  --text:#e8f0f7; --muted:#8da2b7; --accent:#45d0bd; --warn:#efc46b;
  --danger:#ff7f88; --blue:#7db8ff;
}
.stApp{background:var(--bg);color:var(--text)}
[data-testid="stHeader"]{background:var(--bg)}
.block-container{max-width:1600px;padding:1.1rem 2rem 3rem}
section[data-testid="stSidebar"]{background:#081522;border-right:1px solid var(--line)}
section[data-testid="stSidebar"] .block-container{padding-top:1.2rem}
h1,h2,h3,h4{color:var(--text)}
.small,.muted{color:var(--muted)}
.eyebrow{color:var(--accent);font-size:.66rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase}
.hero{background:linear-gradient(135deg,#0e2236,#091522);border:1px solid var(--line);
      border-radius:16px;padding:24px 28px;margin-bottom:14px}
.hero h1{margin:.25rem 0 .35rem;font-size:2rem;letter-spacing:-.03em}
.hero p{margin:0;color:var(--muted)}
.banner{border:1px solid #695625;background:#211c0f;border-radius:10px;padding:11px 14px;color:#e9d38f}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:15px 17px}
.metric-label{font-size:.63rem;color:var(--muted);text-transform:uppercase;letter-spacing:.11em}
.metric-value{font-size:1.55rem;font-weight:800;line-height:1.2;margin-top:7px}
.metric-note{font-size:.68rem;color:var(--muted);margin-top:5px}
.section{margin:21px 0 9px 1px;color:var(--accent);font-size:.67rem;font-weight:800;
         letter-spacing:.15em;text-transform:uppercase}
.state-card{background:linear-gradient(135deg,#0b2330,#0b1926);border:1px solid #28515c;
            border-radius:14px;padding:18px 20px}
.state{font-size:1.8rem;font-weight:900;letter-spacing:.02em;margin:.15rem 0}
.badge{display:inline-block;border-radius:999px;padding:4px 9px;font-size:.61rem;font-weight:850;letter-spacing:.1em}
.ok{color:#6de0cf;background:#0d2b29;border:1px solid #27675e}
.watch{color:#f0c86f;background:#2b2411;border:1px solid #6a5527}
.alert{color:#ff8991;background:#32171b;border:1px solid #71353c}
.neutral{color:#a9bdd0;background:#172534;border:1px solid #30475d}
.pipeline{display:flex;align-items:center;gap:7px;flex-wrap:wrap;background:var(--surface);
          border:1px solid var(--line);border-radius:12px;padding:14px}
.node{padding:7px 10px;background:#101f30;border:1px solid #294057;border-radius:7px;
      font-size:.69rem;color:#d8e5ef}
.arrow{color:#557088}
.trace{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.72rem;color:#a9bdd0}
div[data-testid="stMetric"]{background:var(--surface);border:1px solid var(--line);padding:10px 13px;border-radius:10px}
.stTabs [data-baseweb="tab-list"]{gap:4px;background:#091522;padding:4px;border-radius:10px}
.stTabs [data-baseweb="tab"]{color:#8da2b7;font-size:.75rem}
.stTabs [aria-selected="true"]{color:#e8f0f7}
hr{border-color:#20354a}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------
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

baseline_size = safe_int(
    summary.get("baseline_size", summary.get("baseline", get_value("baseline_size", 0))), 0
)
calibration_size = safe_int(
    summary.get("calibration_size", summary.get("calibration", get_value("calibration_size", 0))), 0
)
test_size = safe_int(
    summary.get("test_size", summary.get("test", get_value("test_size", 0))), 0
)

if state in {"NOMINAL", "NORMAL"}:
    state_cls = "ok"
elif any(x in state for x in ("ANOM", "ALERT", "FAIL")):
    state_cls = "alert"
else:
    state_cls = "watch"

source_label = "SIMULATION" if source in {"SIMULATED", "DEMO"} else source

# ---------------------------------------------------------------------------
# Sidebar: system identity + data maturity
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="eyebrow">SCAN A / DIGITAL TWIN</div>', unsafe_allow_html=True)
    st.markdown("### Engineering Console")
    st.markdown('<div class="small">Condition assessment and multivariate surveillance</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown("**DATA PROVENANCE**")
    st.markdown(
        f'<span class="badge {"watch" if source_label == "SIMULATION" else "ok"}">'
        f'{source_label}</span>',
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

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="hero">'
    '<div class="eyebrow">ENGINEERING MONITORING CONSOLE · V5</div>'
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

# ---------------------------------------------------------------------------
# Command strip
# ---------------------------------------------------------------------------
st.markdown('<div class="section">System state</div>', unsafe_allow_html=True)
cols = st.columns([1.45, 1, 1, 1, 1, 1])
cols[0].markdown(
    f'<div class="state-card"><div class="eyebrow">DIGITAL TWIN STATE</div>'
    f'<div class="state">{state}</div>'
    f'<span class="badge {state_cls}">{state}</span>'
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
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main condition panel
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "Command Center",
    "Acquisition Trace",
    "Digital Signature",
    "Multivariate Monitoring",
    "Drift Surveillance",
    "Validation",
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
            f'<div class="trace">SOURCE  {source_label}<br>'
            f'DRIFT   {drift}<br>'
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
            (c1, "State", "state", None),
            (c2, "Mahalanobis D", "mahalanobis_distance", 4),
            (c3, "Mahalanobis D²", "mahalanobis_squared", 4),
            (c4, "Quality / conformity", "quality_score", 2),
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
            stage = (
                "BASELINE" if idx < baseline_size else
                "CALIBRATION" if idx < baseline_size + calibration_size else
                "TEST"
            )
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

        # Baseline statistics are calculated only from the existing baseline partition.
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
                    records.append({
                        "feature": n,
                        "value": value,
                        "baseline_mean": mu,
                        "baseline_std": sd,
                        "deviation_sigma": z,
                    })
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
                'direct physical measurement.</p>'
                '<div class="trace">D² = D × D</div></div>',
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
        ("SCAM A Digital Signature", "PENDING", "Requires real-system characterization and repeatability."),
        ("Physical baseline / thresholds", "PENDING", "Must be established from validated real SCAN A observations."),
        ("Controlled perturbation study", "PENDING", "Only when safe, authorized and experimentally documented."),
    ]
    for name, status, note in checks:
        cls = "ok" if status == "COMPLETE" else "watch"
        st.markdown(
            f'<div class="card" style="margin-bottom:8px">'
            f'<span class="badge {cls}">{status}</span> <b>{name}</b>'
            f'<div class="small" style="margin-top:5px">{note}</div></div>',
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
    "SCAN A Digital Twin V5 · Presentation layer only · Engineering monitoring console · "
    "Not clinical diagnosis, not automatic proof of hardware failure, not RUL."
)
