"""Interactive research workbench. Run: streamlit run app.py"""

import json
from dataclasses import asdict

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve

from fluxfraud.config import Config
from fluxfraud.evaluation import evaluate
from fluxfraud.features import build_features
from fluxfraud.simulator import simulate

st.set_page_config(page_title="FluxFraud | Stochastic laboratory", page_icon="◈", layout="wide")
st.markdown(
    """<style>
.stApp {background:#0b1020;color:#e8edf7}
[data-testid="stSidebar"] {background:#121a2d}
h1 {letter-spacing:-0.045em} h3 {color:#8be2d0}
[data-testid="stMetric"] {background:#151f34;border:1px solid #293652;border-radius:12px;padding:16px}
</style>""",
    unsafe_allow_html=True,
)
st.caption("FLUXFRAUD  /  COMPUTATIONAL PHYSICS × FINANCIAL DATA SCIENCE")
st.title("Observe the system. Challenge the detector.")
st.write(
    "A controlled stochastic economy for investigating behaviour, fraud and distribution shift."
)
st.sidebar.title("Experiment controls")
with st.sidebar.form("controls"):
    seed = st.number_input("Random seed", 0, 1000000, 42)
    customers = st.slider("Customers", 30, 300, 100, 10)
    days = st.slider("Simulation days", 15, 90, 30, 5)
    eta = st.slider(
        "Branching ratio η",
        0.0,
        0.85,
        0.35,
        0.05,
        help="Expected offspring per event. Higher values create longer bursts.",
    )
    temperature = st.slider(
        "Network temperature T",
        0.2,
        3.0,
        0.8,
        0.1,
        help="Low temperature concentrates recipient choice around preferences.",
    )
    drift = st.slider(
        "Shift strength",
        0.0,
        2.0,
        1.0,
        0.1,
        help="Changes legitimate spending, compromise frequency and fraud camouflage.",
    )
    budget = st.slider("Daily review capacity", 5, 100, 20, 5)
    run = st.form_submit_button("Run experiment", type="primary", width="stretch")
st.sidebar.caption("All data are synthetic. Results describe this model, not any bank's customers.")


@st.cache_data(max_entries=3, show_spinner=False)
def run_world(config):
    sim = simulate(config)
    features = build_features(sim.transactions)
    results, predictions, policies = evaluate(sim.transactions, features, config)
    return sim, results, predictions, policies


if run:
    config = Config(
        seed=seed,
        customers=customers,
        days=days,
        branching_ratio=eta,
        temperature=temperature,
        drift_strength=drift,
        alert_budget=budget,
    )
    try:
        with st.spinner("Simulating the economy and testing six detection strategies…"):
            st.session_state["world"] = (config, *run_world(config))
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))
if "world" not in st.session_state:
    st.info(
        "Choose the experiment controls, then select Run experiment. A run typically takes seconds to a minute."
    )
    st.latex(r"\lambda_i(t)=\mu_i(t)+\sum_{t_k^i<t}\eta\beta e^{-\beta(t-t_k^i)}")
    st.markdown(
        "**Three questions to investigate**\n\n1. Does burst memory improve detection?\n2. Does a changing transfer network reveal hidden coordination?\n3. What survives when fraud starts to resemble ordinary behaviour?"
    )
    st.stop()
config, sim, results, predictions, policies = st.session_state["world"]
tx = sim.transactions
st.caption(
    f"Displayed run: seed {config.seed} · {config.customers} customers · {config.days} days · η={config.branching_ratio:.2f} · T={config.temperature:.1f} · shift={config.drift_strength:.1f}"
)
a, b, c, d = st.columns(4)
a.metric("Transaction attempts", f"{len(tx):,}")
b.metric("Fraud prevalence", f"{tx.is_fraud.mean():.2%}")
c.metric("Conserved balance", f"{sim.accounts.final_balance_cents.sum() / 100:,.0f}")
d.metric("Shift begins", f"Day {config.days * config.drift_fraction:.1f}")
overview, detection, network, mathematics = st.tabs(
    ["Dynamics", "Detection under shift", "Transfer network", "Model & downloads"]
)
with overview:
    hourly = (
        tx.assign(hour=np.floor(tx.time).astype(int))
        .groupby("hour")
        .agg(attempts=("is_fraud", "size"), frauds=("is_fraud", "sum"))
        .reset_index()
    )
    fig = px.line(
        hourly, x="hour", y=["attempts", "frauds"], color_discrete_sequence=["#8be2d0", "#ff9f80"]
    )
    fig.add_vline(x=config.horizon * config.drift_fraction, line_dash="dash", line_color="#dbb8ff")
    fig.update_layout(
        template="plotly_dark", title="Emergent transaction activity", legend_title_text=""
    )
    st.plotly_chart(fig, width="stretch")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.line(
                sim.states,
                x="hour",
                y="compromised",
                title="Hidden compromised population (oracle view)",
                template="plotly_dark",
            ),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            px.histogram(
                tx.assign(log_amount=np.log10(tx.amount_cents / 100)),
                x="log_amount",
                color="is_fraud",
                nbins=55,
                barmode="overlay",
                title="Overlapping amount distributions",
                template="plotly_dark",
            ),
            width="stretch",
        )
    st.caption(
        "Hidden states and labels are shown for scientific diagnosis only. They are excluded from detector inputs."
    )
with detection:
    metric = st.selectbox(
        "Compare",
        [
            "average_precision",
            "precision_at_daily_budget",
            "recall_at_daily_budget",
            "settled_fraud_value_capture",
        ],
    )
    fig = px.bar(
        results[results.period != "all_test"],
        x="model",
        y=metric,
        color="period",
        barmode="group",
        template="plotly_dark",
        color_discrete_sequence=["#8be2d0", "#ff9f80"],
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Daily top-K is retrospective batch triage. Fixed validation thresholds are a separate online-compatible policy; ties can exceed its target alert rate."
    )
    fig = go.Figure()
    for name in policies:
        precision, recall, _ = precision_recall_curve(predictions.is_fraud, predictions[name])
        fig.add_trace(go.Scatter(x=recall, y=precision, name=name, mode="lines"))
    fig.update_layout(
        title="Held-out precision–recall curves",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_dark",
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(results, width="stretch", hide_index=True)
with network:
    st.write(
        "An exploratory view of observed transfers up to the selected day. Colours indicate synthetic communities, not predicted risk."
    )
    day = st.slider("Network cutoff day", 1, config.days, min(15, config.days))
    edges = tx[tx.time < day * 24].groupby(["sender", "receiver"]).size().nlargest(160)
    theta = np.linspace(0, 2 * np.pi, config.customers, endpoint=False)
    xs, ys = np.cos(theta), np.sin(theta)
    ex, ey = [], []
    for i, j in edges.index:
        ex.extend([xs[i], xs[j], None])
        ey.extend([ys[i], ys[j], None])
    fig = go.Figure(
        go.Scatter(
            x=ex, y=ey, mode="lines", line=dict(width=0.7, color="#364762"), hoverinfo="skip"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(size=8, color=sim.accounts.community, colorscale="Teal"),
            text=[f"Account {i}" for i in range(config.customers)],
            hoverinfo="text",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        height=530,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Display limited to the 160 most frequent directed pairs; lines do not show direction. Detector features use the full past event stream."
    )
with mathematics:
    st.latex(r"dX_i=\kappa(m_i-X_i)\,dt+\sigma\,dW_i")
    st.latex(r"P(j\mid i,\mathcal H_t)=\frac{\exp[-E_{ij}(t)/T]}{\sum_{k\ne i}\exp[-E_{ik}(t)/T]}")
    st.write(
        "OU relaxation determines spending; Hawkes memory determines bursts; Gibbs weights govern recipient choice. See README.md for equations, units, assumptions and experimental limits."
    )
    st.download_button(
        "Download exact configuration", json.dumps(asdict(config), indent=2), "config.json"
    )
    st.download_button("Download evaluation", results.to_csv(index=False), "metrics.csv")
    st.download_button(
        "Download synthetic transactions", tx.to_csv(index=False), "transactions.csv"
    )
