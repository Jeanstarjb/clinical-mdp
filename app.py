"""
Interactive demo for the Clinical MDP engine. Imports the backend
services directly -- this app never reimplements the solver, so it
can't drift from the version that's actually correctness-tested.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

import pandas as pd
import streamlit as st

from services import clinical_scenario
from services.simulation_service import evaluate_optimal_vs_baseline

st.set_page_config(page_title="Clinical MDP Demo", page_icon="🩺", layout="wide")

st.title("🩺 Clinical Decision Support via Markov Decision Processes")
st.caption(
    "A treatment-escalation policy computed by value iteration on a diabetes-management "
    "scenario, then checked against 1000s of Monte Carlo rollouts. The reward structure uses "
    "real published decrement coefficients "
    "([Oh et al. 2021, Scientific Reports](https://doi.org/10.1038/s41598-021-86419-4)); "
    "**transition probabilities remain synthetic** (no public source publishes these for a "
    "multi-action decision problem -- see the README). This demonstrates a decision "
    "algorithm, not medical guidance."
)

with st.sidebar:
    st.header("Settings")
    gamma = st.slider("Discount factor (γ)", 0.5, 0.99, 0.9, step=0.01)
    n_simulations = st.slider("Monte Carlo simulations", 500, 10000, 3000, step=500)
    start_state = st.selectbox("Starting state for comparison", clinical_scenario.STATES, index=2)
    run = st.button("Solve", type="primary")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Scenario")
    st.markdown(f"**States:** {', '.join(clinical_scenario.STATES)}")
    st.markdown(f"**Actions:** {', '.join(clinical_scenario.ACTIONS)}")
    reward_df = pd.DataFrame(clinical_scenario.REWARDS).T
    st.markdown("**Reward table** (state utility − treatment burden)")
    st.dataframe(reward_df.style.format("{:.1f}"), use_container_width=True)

if run:
    with st.spinner("Solving via value iteration and running Monte Carlo evaluation..."):
        result = evaluate_optimal_vs_baseline(
            start_state=start_state, n_simulations=n_simulations, gamma=gamma
        )

    with col2:
        st.subheader("Optimal policy")
        policy_df = pd.DataFrame(
            [(s, result['optimal_policy'][s], result['state_values'][s]) for s in clinical_scenario.STATES],
            columns=["State", "Optimal action", "Value V(s)"],
        )
        st.dataframe(policy_df.style.format({"Value V(s)": "{:.2f}"}), use_container_width=True)
        st.metric("Bellman residual (correctness check, should be ~0)", f"{result['bellman_residual']:.2e}")

    st.divider()
    st.subheader(f"Optimal vs. baseline policy — {n_simulations} Monte Carlo rollouts from '{start_state}'")

    c1, c2, c3 = st.columns(3)
    opt = result['comparison']['optimal']
    base = result['comparison']['baseline_monotherapy']

    with c1:
        st.metric("Optimal policy — mean discounted reward", f"{opt['mean_reward']:.2f}", f"± {opt['std_error']:.3f}")
    with c2:
        st.metric("Baseline (always Monotherapy)", f"{base['mean_reward']:.2f}", f"± {base['std_error']:.3f}")
    with c3:
        st.metric("Improvement", f"{result['improvement_over_baseline']:.2f}")

    chart_df = pd.DataFrame({
        "policy": ["optimal", "baseline_monotherapy"],
        "mean_reward": [opt['mean_reward'], base['mean_reward']],
    }).set_index("policy")
    st.bar_chart(chart_df)

    st.caption(
        "Both numbers come from simulating the same MDP thousands of times under each policy — "
        "not assumed. The optimal policy adapts treatment intensity to the patient's state; the "
        "baseline applies the same treatment regardless of state."
    )
else:
    st.info("Adjust settings in the sidebar and click **Solve** to compute the optimal policy.")
