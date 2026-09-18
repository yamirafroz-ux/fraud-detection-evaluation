# FluxFraud

A physics-inspired simulation of financial transactions and fraud. It generates synthetic customers, payments and known fraud labels, then compares how machine-learning models perform when customer and fraud behaviour change.

## The maths behind it

- **Hawkes processes** generate bursts of transactions: one payment temporarily increases the chance of another.
- **Markov chains** move customers between ordinary, travelling and high-spending states, with a separate compromise/recovery process.
- **Ornstein–Uhlenbeck dynamics** make a customer's log-spending tendency fluctuate around a changing baseline. A higher spending scale means larger typical attempted payments, not a higher balance or spending limit.
- **Gibbs probabilities** choose recipients using community preferences, previous interactions and hidden fraud-related preferences.
- **Conservation of money** keeps total funds constant: successful transfers debit one account and credit another. Insufficient funds cause a decline.
- **Monte Carlo experiments** repeat the simulation with different seeds to measure variation in results.

The models include logistic regression, gradient boosting, Isolation Forest and a simple behavioural rule. Training uses earlier transactions; testing uses later transactions, including a scheduled change in the simulated environment. Features use only information available at the transaction time.

This is a synthetic research project. Its parameters are not calibrated to a real bank, and its results do not establish real-world fraud-detection performance.

## Run locally

Requires Python 3.11 or newer. From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[app,dev]'
streamlit run app.py
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

Open the address printed by Streamlit. Play the transaction animation, inspect transfer loops and compare model predictions. The animation replays a generated simulation; it does not train models as it plays.

## More information

- [Usage and parameter guide](HOW_TO_USE.txt)
- [Mathematical guide](output/pdf/FluxFraud_Mathematical_Foundations.pdf): derivations, worked examples and references.
- [Code companion](output/pdf/FluxFraud_Mathematics_to_Code.pdf): how the mathematics maps to the implementation.
- [Literature](docs/LITERATURE.md) and [experimental results](reports/RESULTS.md).

Simulation and modelling code lives in `src/fluxfraud/`. The application starts in `app.py`, and default parameters are in `configs/default.json`.
