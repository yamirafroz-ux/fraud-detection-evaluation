"""A simple animated view of the stochastic laboratory. Run: streamlit run app.py."""

import json
from dataclasses import asdict

import streamlit as st
import streamlit.components.v1 as components

from fluxfraud.config import Config
from fluxfraud.evaluation import evaluate
from fluxfraud.features import build_features
from fluxfraud.playback import replay_html
from fluxfraud.simulator import simulate

st.set_page_config(page_title="FluxFraud · Watch the money move", page_icon="◈", layout="wide")
st.markdown(
    """<style>.stApp{background:#0c1220;color:#e8edf6}.block-container{padding-top:2rem;max-width:1250px}header[data-testid="stHeader"]{background:#0c1220}h1{letter-spacing:-.04em}</style>""",
    unsafe_allow_html=True,
)
st.title("Watch the money move.")
st.caption(
    "Accounts light up with each transfer. Follow the loops. See which models catch the fraud."
)


@st.cache_data(max_entries=3, show_spinner=False)
def run_world(config):
    sim = simulate(config)
    results, predictions, policies = evaluate(
        sim.transactions, build_features(sim.transactions), config
    )
    return sim, results, predictions, policies


with st.expander("New simulation · change duration or number of accounts"):
    with st.form("settings"):
        a, b, c = st.columns(3)
        customers = a.selectbox("Accounts", [36, 64, 100, 144], index=1)
        days = b.selectbox("Duration", [30, 60, 90], format_func=lambda d: f"{d} days")
        seed = c.number_input("Random seed", min_value=0, value=42, step=1)
        run = st.form_submit_button("Create new simulation", type="primary")
if run or "replay_world" not in st.session_state:
    config = Config(customers=customers, days=days, seed=seed)
    try:
        with st.spinner("Preparing your world and training the models…"):
            st.session_state["replay_world"] = (config, *run_world(config))
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))
        st.stop()
config, sim, results, predictions, policies = st.session_state["replay_world"]
components.html(replay_html(config, sim, predictions, policies), height=1140, scrolling=True)
with st.expander("Download this simulation"):
    st.download_button("Settings", json.dumps(asdict(config), indent=2), "config.json")
    st.download_button(
        "Model results · entire test period", results.to_csv(index=False), "metrics.csv"
    )
    st.download_button("Transactions", sim.transactions.to_csv(index=False), "transactions.csv")
st.caption(
    f"Synthetic data · {config.customers} accounts · {config.days} days · seed {config.seed}. Loops show one observed 2- or 3-transfer cycle per closing event, not proof of fraud."
)
