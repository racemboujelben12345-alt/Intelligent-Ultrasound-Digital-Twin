"""Intelligent Ultrasound Digital Twin — Engineering Monitoring Workstation.

Presentation layer only. The scientific pipeline and artifacts remain unchanged.
"""

from __future__ import annotations

from pathlib import Path
import json
import math

import pandas as pd
import streamlit as st
\nfrom src.digital_twin.health import assess_twin_health\n

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Ultrasound | Engineering Workstation",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# VISUAL SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    :root{
      --bg:#e7edf1;
      --surface:#fbfcfd;
      --surface2:#eef3f6;
      --line:#c7d3dc;
      --text:#1b2b39;
      --muted:#607484;
      --navy:#18344b;
      --blue:#1769aa;
      --teal:#087f83;
      --warn:#a86d09;
      --danger:#b33b47;
      --shadow:0 1px 4px rgba(25,45,60,.08);
    }

    .stApp{
      background:var(--bg);
      color:var(--text);
    }

    [data-testid="stHeader"]{
      display:none;
    }

    .block-container{
      max-width:1640px;
      padding:0 2rem 3rem;
    }

    section[data-testid="stSidebar"]{
      background:#f4f7f9;
      border-right:1px solid var(--line);
    }

    section[data-testid="stSidebar"] .block-container{
      padding-top:1rem;
    }

    h1,h2,h3,h4{
      color:var(--text);
    }

    .small,
    .muted{
      color:var(--muted);
    }

    .eyebrow{
      color:var(--blue);
      font-size:.62rem;
      font-weight:800;
      letter-spacing:.15em;
      text-transform:uppercase;
    }

    .appbar{
      background:var(--navy);
      color:#f4f8fa;
      margin:0 -2rem 18px;
      padding:14px 2rem;
      display:flex;
      justify-content:space-between;
      align-items:center;
      border-bottom:1px solid #47667b;
    }

    .appbar-title{
      font-size:.98rem;
      font-weight:850;
      letter-spacing:.06em;
    }

    .appbar-sub{
      font-size:.68rem;
      color:#b9c9d4;
      margin-top:4px;
      letter-spacing:.03em;
    }

    .appbar-status{
      font-size:.62rem;
      font-weight:850;
      letter-spacing:.08em;
      padding:6px 9px;
      border:1px solid #718b9d;
      border-radius:3px;
    }

    .hero{
      background:var(--surface);
      border:1px solid var(--line);
      border-left:4px solid var(--blue);
      border-radius:5px;
      padding:17px 21px;
      margin-bottom:10px;
      box-shadow:var(--shadow);
    }

    .hero h1{
      margin:.18rem 0 .25rem;
      font-size:1.72rem;
      letter-spacing:-.018em;
    }

    .hero p{
      margin:0;
      color:var(--muted);
      font-size:.82rem;
    }

    .banner{
      border:1px solid #dfc98f;
      background:#fff8e5;
      border-radius:4px;
      padding:9px 12px;
      color:#73530a;
      font-size:.78rem;
    }

    .card{
      background:var(--surface);
      border:1px solid var(--line);
      border-radius:5px;
      padding:12px 14px;
      box-shadow:var(--shadow);
    }

    .metric-label{
      font-size:.59rem;
      color:var(--muted);
      text-transform:uppercase;
      letter-spacing:.1em;
    }

    .metric-value{
      font-size:1.35rem;
      font-weight:800;
      line-height:1.15;
      margin-top:6px;
      color:var(--text);
    }

    .metric-note{
      font-size:.64rem;
      color:var(--muted);
      margin-top:4px;
    }

    .section{
      margin:18px 0 7px 1px;
      color:#405b6e;
      font-size:.62rem;
      font-weight:850;
      letter-spacing:.16em;
      text-transform:uppercase;
      border-bottom:1px solid #cbd6de;
      padding-bottom:5px;
    }

    .state-card{
      background:#f9fbfc;
      border:1px solid #b8ccd8;
      border-left:4px solid var(--teal);
      border-radius:5px;
      padding:14px 17px;
      box-shadow:var(--shadow);
    }

    .state{
      font-size:1.62rem;
      font-weight:900;
      letter-spacing:.01em;
      margin:.12rem 0;
    }

    .badge{
      display:inline-block;
      border-radius:3px;
      padding:3px 7px;
      font-size:.57rem;
      font-weight:850;
      letter-spacing:.08em;
    }

    .ok{
      color:#17634f;
      background:#e6f3ee;
      border:1px solid #a8d2c2;
    }

    .watch{
      color:#80590a;
      background:#fff1d0;
      border:1px solid #e4ca82;
    }

    .alert{
      color:#96313c;
      background:#fbe9eb;
      border:1px solid #dfadb4;
    }

    .neutral{
      color:#4f6675;
      background:#edf2f5;
      border:1px solid #c9d5dd;
    }

    .pipeline{
      display:flex;
      align-items:center;
      gap:6px;
      flex-wrap:wrap;
      background:#f8fafb;
      border:1px solid var(--line);
      border-radius:4px;
      padding:10px;
    }

    .node{
      padding:6px 9px;
      background:#edf3f6;
      border:1px solid #c9d7df;
      border-radius:3px;
      font-size:.64rem;
      color:#29465b;
      font-weight:700;
    }

    .arrow{
      color:#7f95a4;
    }

    .trace{
      font-family:ui-monospace,SFMono-Regular,Consolas,monospace;
      font-size:.68rem;
      color:#526b7c;
    }

    div[data-testid="stMetric"]{
      background:var(--surface);
      border:1px solid var(--line);
      padding:8px 11px;
      border-radius:4px;
      box-shadow:none;
    }

    .stTabs [data-baseweb="tab-list"]{
      gap:0;
      background:#dbe4ea;
      padding:3px;
      border:1px solid #c4d0d8;
      border-radius:4px;
    }

    .stTabs [data-baseweb="tab"]{
      color:#526b7c;
      font-size:.7rem;
      border-radius:3px;
    }

    .stTabs [aria-selected="true"]{
      color:var(--blue);
      font-weight:800;
      background:#f9fbfc;
    }

    hr{
      border-color:var(--line);
    }

    [data-testid="stDataFrame"]{
      border:1px solid var(--line);
      border-radius:4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADERS
# ============================================================

def load_json(name: str) -> dict:
    path = OUT / name

    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_csv(name: str) -> pd.DataFrame:
    path = OUT / name

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


# ============================================================
# LOAD PIPELINE ARTIFACTS
# ============================================================

summary = load_json("summary.json")
states = load_csv("twin_states.csv")
features = load_csv("features.csv")
report_path = OUT / "report.md"


if states.empty and not summary:
    st.error("No pipeline artifacts found. Run: python main.py")
    st.stop()


latest = states.iloc[-1] if not states.empty else pd.Series(dtype=object)


# ============================================================
# HELPERS
# ============================================================

def get_value(key: str, fallback="—"):
    value = latest.get(key, summary.get(key, fallback))

    try:
        if pd.isna(value):
            return fallback
    except Exception:
        pass

    return value


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


# ============================================================
# CURRENT SYSTEM STATE
# ============================================================

source = str(get_value("source", "unknown")).upper()
state = str(get_value("state", "UNKNOWN")).upper()
acq_id = str(get_value("acquisition_id", "—"))
drift = str(get_value("drift_status", "—")).upper()

n_acq = safe_int(
    get_value("n_acquisitions", len(states)),
    len(states),
)

baseline_size = safe_int(
    summary.get(
        "baseline_size",
        summary.get(
            "baseline",
            get_value("baseline_size", 0),
        ),
    ),
    0,
)

calibration_size = safe_int(
    summary.get(
        "calibration_size",
        summary.get(
            "calibration",
            get_value("calibration_size", 0),
        ),
    ),
    0,
)

test_size = safe_int(
    summary.get(
        "test_size",
        summary.get(
            "test",
            get_value("test_size", 0),
        ),
    ),
    0,
)


if state in {"NOMINAL", "NORMAL"}:
    state_cls = "ok"
elif any(x in state for x in ("ANOM", "ALERT", "FAIL")):
    state_cls = "alert"
else:
    state_cls = "watch"


source_label = (
    "SIMULATION"
    if source in {"SIMULATED", "DEMO"}
    else source
)


# ============================================================
# DIGITAL TWIN HEALTH & EVIDENCE CONFIDENCE
# ============================================================

try:
    _quality = float(get_value("quality_score", 0.0))
    _distance = float(get_value("mahalanobis_distance", 0.0))
except (TypeError, ValueError):
    _quality = 0.0
    _distance = 0.0

_feature_columns = [
    name for name in features.columns
    if name not in {"acquisition_id", "acquisition_index", "source"}
    and pd.api.types.is_numeric_dtype(features[name])
]
_expected_features = max(len(_feature_columns), 1)

if source in {"EXPERIMENTAL", "SCAN_A"}:
    _provenance_score = 100.0
elif source == "PUBLIC":
    _provenance_score = 80.0
else:
    _provenance_score = 60.0

_health = assess_twin_health(
    quality_score=_quality,
    mahalanobis_distance=max(_distance, 0.0),
    baseline_observations=baseline_size,
    feature_count=len(_feature_columns),
    expected_feature_count=_expected_features,
    provenance_score=_provenance_score,
)

health_state_cls = (
    "ok" if _health.health_state == "NOMINAL"
    else "alert" if _health.health_state == "HIGH_DEVIATION"
    else "watch"
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="eyebrow">ultrasound system / DIGITAL TWIN</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Engineering Workstation")

    st.markdown(
        '<div class="small">'
        'Condition assessment and multivariate surveillance'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("**DATA PROVENANCE**")

    badge_class = (
        "watch"
        if source_label == "SIMULATION"
        else "ok"
    )

    st.markdown(
        f'<span class="badge {badge_class}">{source_label}</span>',
        unsafe_allow_html=True,
    )

    st.caption("Current artifacts")

    st.markdown("**VALIDATION MATURITY**")
    st.markdown("Software pipeline · **IMPLEMENTED**")
    st.markdown("Simulation · **IMPLEMENTED**")
    st.markdown("Public benchmark · **IMPLEMENTED / DATA-DEPENDENT**")
    st.markdown("Real ultrasound · **PENDING**")
    st.markdown("Physical baseline · **PENDING**")

    st.divider()

    st.markdown("**PIPELINE**")

    st.caption(
        "Acquisition → Signature → Quality → Baseline → AI → Fusion → "
        "Twin State → Drift → Prediction → V&V"
    )

    st.caption(
        "Presentation layer does not modify the scientific engine."
    )


# ============================================================
# APPLICATION HEADER
# ============================================================

st.markdown(
    '<div class="appbar">'
    '<div>'
    '<div class="appbar-title">'
    'ultrasound system · INTELLIGENT ULTRASOUND DIGITAL TWIN'
    '</div>'
    '<div class="appbar-sub">'
    'Biomedical Engineering · Condition Monitoring'
    '</div>'
    '</div>'
    f'<div class="appbar-status">● {state}</div>'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="hero">'
    '<div class="eyebrow">SYSTEM OVERVIEW</div>'
    '<h1>Engineering condition assessment</h1>'
    '<p>'
    'Digital Signature · Multivariate monitoring · '
    'Drift surveillance · Validation maturity'
    '</p>'
    '</div>',
    unsafe_allow_html=True,
)


if source_label == "SIMULATION":

    st.markdown(
        '<div class="banner">'
        '<b>SIMULATION MODE</b> · '
        'Synthetic observations validate the software workflow only. '
        'They do not establish a physical ultrasound system fault, threshold, '
        'or hardware-health percentage.'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# SYSTEM STATE / ACQUISITION CONTEXT
# ============================================================

st.markdown(
    '<div class="section">'
    'System state · acquisition context'
    '</div>',
    unsafe_allow_html=True,
)

cols = st.columns([1.45, 1, 1, 1, 1, 1])


cols[0].markdown(
    f'<div class="state-card">'
    f'<div class="eyebrow">DIGITAL TWIN STATE</div>'
    f'<div class="state">{state}</div>'
    f'<span class="badge {state_cls}">{state}</span>'
    f'<div class="small" style="margin-top:8px">'
    f'Latest acquisition · {acq_id}'
    f'</div>'
    f'</div>',
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
        f'<div class="card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-note">{note}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# CURRENT CONDITION ASSESSMENT
# ============================================================

st.markdown(
    '<div class="section">'
    'Current condition assessment'
    '</div>',
    unsafe_allow_html=True,
)

a, b, c, d = st.columns(4)

a.metric(
    "Mahalanobis D",
    fmt("mahalanobis_distance", 4),
    help=(
        "Multivariate statistical distance from "
        "the learned feature distribution."
    ),
)

b.metric(
    "Mahalanobis D²",
    fmt("mahalanobis_squared", 4),
    help="Squared Mahalanobis distance.",
)

c.metric(
    "Quality / conformity",
    fmt("quality_score", 2),
    help=(
        "Pipeline score; not a device-health percentage."
    ),
)

d.metric(
    "Drift status",
    drift,
    help=(
        "Current drift classification produced "
        "by the existing pipeline."
    ),
)

st.markdown(
    '<div class="section">AI & intelligence evidence</div>',
    unsafe_allow_html=True,
)

ai1, ai2, ai3, ai4 = st.columns(4)

ai1.metric(
    "AI anomaly evidence",
    fmt("ai_anomaly_score", 3),
    help="Unsupervised AI evidence; not a physical-failure probability.",
)

ai2.metric(
    "AI confidence",
    fmt("ai_confidence", 3),
    help="Ensemble confidence derived from model agreement and score dispersion.",
)

ai3.metric(
    "Fused intelligence",
    fmt("intelligence_fused_score", 3),
    help="Transparent fusion of statistical, AI and quality evidence.",
)

ai4.metric(
    "Fusion state",
    get_value("intelligence_state", "—"),
    help="Engineering evidence state, not a clinical or hardware-failure diagnosis.",
)




# ============================================================
# TWIN HEALTH PANEL
# ============================================================

st.markdown(
    '<div class="section">Twin health · evidence confidence</div>',
    unsafe_allow_html=True,
)

h1, h2, h3, h4 = st.columns(4)

h1.metric(
    "Twin Health Index",
    f"{_health.health_index:.1f}/100",
    help="Engineering condition indicator; not a hardware-health percentage.",
)
h2.metric(
    "Evidence Confidence",
    f"{_health.confidence_score:.1f}/100",
    help="Evidence maturity score; not a probability of correctness or failure.",
)
h3.metric(
    "Quality Component",
    f"{_health.quality_component:.1f}/100",
)
h4.metric(
    "Statistical Component",
    f"{_health.anomaly_component:.1f}/100",
)

st.markdown(
    f'<div class="state-card">'
    f'<div class="eyebrow">TWIN HEALTH STATE</div>'
    f'<div class="state">{_health.health_state}</div>'
    f'<span class="badge {health_state_cls}">{_health.health_state}</span>'
    f'<div class="small" style="margin-top:8px">'
    f'Dominant evidence · {", ".join(_health.dominant_evidence)}'
    f'</div>'
    f'<div class="small" style="margin-top:5px">'
    f'{_health.interpretation}'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ============================================================
# DIGITAL TWIN CHAIN
# ============================================================

st.markdown(
    '<div class="section">Digital twin chain</div>',
    unsafe_allow_html=True,
)

nodes = [
    "Acquisition",
    "Digital Signature",
    "Statistical Baseline",
    "Anomaly Detection",
    "Twin State",
    "Drift",
    "Trend / Prediction",
]

html = '<div class="pipeline">'

for i, node in enumerate(nodes):

    html += f'<span class="node">{node}</span>'

    if i < len(nodes) - 1:
        html += '<span class="arrow">→</span>'

html += "</div>"

st.markdown(
    html,
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "Command Center",
        "Acquisition Trace",
        "Digital Signature",
        "Multivariate Monitoring",
        "Drift Surveillance",
        "Validation",
        "Engineering Report",
    ]
)


# ============================================================
# TAB 1 — COMMAND CENTER
# ============================================================

with tabs[0]:

    st.markdown(
        '<div class="section">Operational overview</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.55, 1])

    with left:

        if not states.empty:

            chart_cols = [
                x
                for x in [
                    "mahalanobis_distance",
                    "quality_score",
                ]
                if x in states.columns
            ]

            if chart_cols:

                chart = states[chart_cols].copy()

                if "acquisition_id" in states.columns:
                    chart.index = (
                        states["acquisition_id"]
                        .astype(str)
                    )

                st.line_chart(
                    chart,
                    height=330,
                )

        st.markdown(
            '<div class="small">'
            'Observed trajectory from generated twin states. '
            'No physical acceptance threshold is displayed '
            'unless established from real ultrasound validation.'
            '</div>',
            unsafe_allow_html=True,
        )

    with right:

        st.markdown(
            '<div class="card">'
            '<div class="eyebrow">ENGINEERING INTERPRETATION</div>'
            f'<h3 style="margin:.35rem 0">'
            f'Current state: {state}'
            f'</h3>'
            f'<p class="muted">'
            f'Acquisition <b>{acq_id}</b> is currently classified '
            f'by the existing multivariate pipeline as '
            f'<b>{state}</b>. The statistical distance and drift '
            f'outputs must be interpreted against a validated '
            f'ultrasound baseline.'
            f'</p>'
            f'<div class="trace">'
            f'SOURCE  {source_label}<br>'
            f'DRIFT   {drift}<br>'
            f'FEATURES '
            f'{len([x for x in features.columns if x not in {"acquisition_id", "acquisition_index"}]) if not features.empty else "—"}'
            f'</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            '<div class="card">'
            '<div class="eyebrow">SCIENTIFIC BOUNDARY</div>'
            '<p class="muted">'
            'Public and synthetic ultrasound data support '
            'methodological and software validation. '
            'They must not silently define the physical '
            'Ultrasound Digital Signature, hardware thresholds, '
            'or a diagnosis.'
            '</p>'
            '</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# TAB 2 — ACQUISITION TRACE
# ============================================================

with tabs[1]:

    st.markdown(
        '<div class="section">Acquisition traceability</div>',
        unsafe_allow_html=True,
    )

    if states.empty:

        st.warning(
            "No twin_states.csv artifact."
        )

    else:

        if "acquisition_id" in states.columns:
            ids = (
                states["acquisition_id"]
                .astype(str)
                .tolist()
            )
        else:
            ids = [
                str(i)
                for i in states.index
            ]

        selected = st.selectbox(
            "Acquisition record",
            ids,
            index=len(ids) - 1,
        )

        row = states.iloc[
            ids.index(selected)
        ]

        c1, c2, c3, c4 = st.columns(4)

        display_fields = [
            (
                c1,
                "State",
                "state",
                None,
            ),
            (
                c2,
                "Mahalanobis D",
                "mahalanobis_distance",
                4,
            ),
            (
                c3,
                "Mahalanobis D²",
                "mahalanobis_squared",
                4,
            ),
            (
                c4,
                "Quality / conformity",
                "quality_score",
                2,
            ),
        ]

        for col, label, key, digits in display_fields:

            raw = row.get(key, "—")

            try:

                shown = (
                    f"{float(raw):.{digits}f}"
                    if digits is not None
                    else str(raw)
                )

            except Exception:

                shown = str(raw)

            col.metric(
                label,
                shown,
            )

        st.markdown(
            '<div class="section">Record payload</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(
            pd.DataFrame([row]),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            '<div class="section">Acquisition position</div>',
            unsafe_allow_html=True,
        )

        try:

            idx = ids.index(selected)

            if idx < baseline_size:
                stage = "BASELINE"
            elif idx < baseline_size + calibration_size:
                stage = "CALIBRATION"
            else:
                stage = "TEST"

            st.markdown(
                f'<div class="card">'
                f'<span class="badge neutral">{stage}</span> '
                f'<span class="small">'
                f'Sequence position {idx + 1} / {len(ids)}'
                f'</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        except Exception:
            pass


# ============================================================
# TAB 3 — DIGITAL SIGNATURE
# ============================================================

with tabs[2]:

    st.markdown(
        '<div class="section">'
        'Digital Signature · feature-space inspection'
        '</div>',
        unsafe_allow_html=True,
    )

    if features.empty:

        st.warning(
            "No features.csv artifact."
        )

    else:

        if "acquisition_id" in features.columns:
            ids = (
                features["acquisition_id"]
                .astype(str)
                .tolist()
            )
        else:
            ids = [
                str(i)
                for i in features.index
            ]

        selected = st.selectbox(
            "Signature acquisition",
            ids,
            index=len(ids) - 1,
            key="signature_v5",
        )

        row = features.iloc[
            ids.index(selected)
        ]

        excluded = {
            "acquisition_id",
            "acquisition_index",
            "source",
        }

        names = [
            name
            for name in features.columns
            if name not in excluded
            and pd.api.types.is_numeric_dtype(
                features[name]
            )
        ]

        bdf = (
            features.iloc[:baseline_size]
            if baseline_size > 1
            else features.iloc[:0]
        )

        baseline_stats = {}

        if not bdf.empty:

            for name in names:

                mu = pd.to_numeric(
                    bdf[name],
                    errors="coerce",
                ).mean()

                sd = pd.to_numeric(
                    bdf[name],
                    errors="coerce",
                ).std(ddof=1)

                baseline_stats[name] = (
                    mu,
                    sd,
                )

        if names:

            top = st.columns(4)

            for i, name in enumerate(names[:12]):

                raw = row.get(
                    name,
                    "—",
                )

                try:
                    shown = f"{float(raw):.5g}"
                except Exception:
                    shown = str(raw)

                top[i % 4].metric(
                    name.replace("_", " "),
                    shown,
                )

            st.markdown(
                '<div class="section">'
                'Signature profile'
                '</div>',
                unsafe_allow_html=True,
            )

            profile = pd.DataFrame(
                {
                    "value": [
                        pd.to_numeric(
                            row[n],
                            errors="coerce",
                        )
                        for n in names
                    ]
                },
                index=names,
            )

            st.bar_chart(
                profile,
                height=360,
            )

            if baseline_stats:

                records = []

                for name in names:

                    mu, sd = baseline_stats[name]

                    value = pd.to_numeric(
                        row[name],
                        errors="coerce",
                    )

                    if (
                        sd
                        and not math.isnan(sd)
                    ):
                        z = (
                            value - mu
                        ) / sd
                    else:
                        z = float("nan")

                    records.append(
                        {
                            "feature": name,
                            "value": value,
                            "baseline_mean": mu,
                            "baseline_std": sd,
                            "deviation_sigma": z,
                        }
                    )

                dev = pd.DataFrame(
                    records
                ).sort_values(
                    "deviation_sigma",
                    key=lambda s: s.abs(),
                    ascending=False,
                )

                st.markdown(
                    '<div class="section">'
                    'Deviation from current software baseline partition'
                    '</div>',
                    unsafe_allow_html=True,
                )

                st.dataframe(
                    dev,
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown(
            '<div class="section">'
            'Complete signature matrix'
            '</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(
            features,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# TAB 4 — MULTIVARIATE MONITORING
# ============================================================

with tabs[3]:

    st.markdown(
        '<div class="section">'
        'Multivariate monitoring'
        '</div>',
        unsafe_allow_html=True,
    )

    if states.empty:

        st.warning(
            "No state history available."
        )

    else:

        c1, c2 = st.columns([1.6, 1])

        with c1:

            numeric = [
                x
                for x in [
                    "mahalanobis_distance",
                    "mahalanobis_squared",
                    "quality_score",
                ]
                if x in states.columns
            ]

            if numeric:

                chart = states[numeric].copy()

                if "acquisition_id" in states.columns:
                    chart.index = (
                        states["acquisition_id"]
                        .astype(str)
                    )

                st.line_chart(
                    chart,
                    height=400,
                )

        with c2:

            st.markdown(
                '<div class="card">'
                '<div class="eyebrow">'
                'STATISTICAL MEANING'
                '</div>'
                '<p class="muted">'
                'Mahalanobis distance summarizes multivariate '
                'separation of an observation from the learned '
                'feature distribution. It is a statistical '
                'monitoring quantity, not a direct physical '
                'measurement.'
                '</p>'
                '<div class="trace">'
                'D² = D × D'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            if "state" in states.columns:

                counts = (
                    states["state"]
                    .astype(str)
                    .value_counts()
                    .rename_axis("state")
                    .to_frame("count")
                )

                st.dataframe(
                    counts,
                    use_container_width=True,
                )

        display = [
            x
            for x in [
                "acquisition_id",
                "source",
                "state",
                "mahalanobis_distance",
                "mahalanobis_squared",
                "quality_score",
                "drift_status",
            ]
            if x in states.columns
        ]

        st.dataframe(
            states[display],
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# TAB 5 — DRIFT SURVEILLANCE
# ============================================================

with tabs[4]:

    st.markdown(
        '<div class="section">'
        'Drift surveillance · temporal behavior'
        '</div>',
        unsafe_allow_html=True,
    )

    if states.empty:

        st.warning(
            "No temporal history available."
        )

    else:

        if "drift_status" in states.columns:

            drift_counts = (
                states["drift_status"]
                .astype(str)
                .value_counts()
                .rename_axis("drift_status")
                .to_frame("observations")
            )

            st.dataframe(
                drift_counts,
                use_container_width=True,
                hide_index=True,
            )

        if "quality_score" in states.columns:

            q = states[
                ["quality_score"]
            ].copy()

            if "acquisition_id" in states.columns:
                q.index = (
                    states["acquisition_id"]
                    .astype(str)
                )

            st.markdown(
                '<div class="section">'
                'Quality trajectory'
                '</div>',
                unsafe_allow_html=True,
            )

            st.line_chart(
                q,
                height=300,
            )

        st.markdown(
            '<div class="card">'
            '<div class="eyebrow">'
            'DRIFT INTERPRETATION'
            '</div>'
            '<p class="muted">'
            'Temporal surveillance identifies changes in '
            'the statistical behavior of the observed feature '
            'signature. Persistence and direction are useful '
            'for engineering review, but the current synthetic '
            'workflow cannot be interpreted as proof of a '
            'physical ultrasound-system drift.'
            '</p>'
            '<p class="muted">'
            '<b>'
            'Not RUL · not a failure-date predictor · '
            'not a hardware diagnosis.'
            '</b>'
            '</p>'
            '</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# TAB 6 — VALIDATION
# ============================================================

with tabs[5]:

    st.markdown(
        '<div class="section">'
        'Validation maturity matrix'
        '</div>',
        unsafe_allow_html=True,
    )

    checks = [
        (
            "Software pipeline",
            "COMPLETE",
            "Execution and artifact generation.",
        ),
        (
            "Synthetic simulation",
            "COMPLETE",
            "Controlled workflow and statistical-engine behavior.",
        ),
        (
            "Public ultrasound benchmark",
            "COMPLETE",
            "Methodological/software validation only.",
        ),
        (
            "Real ultrasound acquisition",
            "PENDING",
            "Repeated controlled acquisitions from the physical system.",
        ),
        (
            "Ultrasound Digital Signature",
            "PENDING",
            "Requires real-system characterization and repeatability.",
        ),
        (
            "Physical baseline / thresholds",
            "PENDING",
            "Must be established from validated real ultrasound observations.",
        ),
        (
            "Controlled perturbation study",
            "PENDING",
            "Only when safe, authorized and experimentally documented.",
        ),
    ]

    for name, status, note in checks:

        cls = (
            "ok"
            if status == "COMPLETE"
            else "watch"
        )

        st.markdown(
            f'<div class="card" style="margin-bottom:8px">'
            f'<span class="badge {cls}">{status}</span> '
            f'<b>{name}</b>'
            f'<div class="small" style="margin-top:5px">'
            f'{note}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section">Data governance</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="card">'
        f'<div class="trace">'
        f'CURRENT SOURCE  {source_label}'
        f'</div>'
        '<p class="muted">'
        'Synthetic and public datasets are explicitly separated '
        'from the future physical ultrasound baseline. This prevents '
        'benchmark behavior from being presented as machine-specific '
        'evidence.'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# TAB 7 — ENGINEERING REPORT
# ============================================================

with tabs[6]:

    st.markdown(
        '<div class="section">'
        'Generated engineering report'
        '</div>',
        unsafe_allow_html=True,
    )

    if report_path.exists():

        st.markdown(
            report_path.read_text(
                encoding="utf-8"
            )
        )

    else:

        st.info(
            "No report.md artifact available."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Intelligent Ultrasound Digital Twin · Presentation layer only · "
    "Real ultrasound characterization remains pending. "
    "Not clinical diagnosis, not automatic proof of hardware failure, not RUL."
)