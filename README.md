# Clinical Decision Support via Markov Decision Processes

A Markov Decision Process (MDP) solver applied to an illustrative treatment-escalation scenario, with an exact dynamic-programming solution cross-validated against independent Monte Carlo simulation.

## Overview

Sequential treatment decisions — how aggressively to treat a patient given their current state — are naturally framed as an MDP: a set of states, a set of possible actions, transition probabilities describing how the patient's state responds to each action, and a reward capturing the tradeoff between outcome and treatment burden. Solving the MDP gives the policy (an action for every state) that maximizes expected long-run reward.

This repo solves a small, hand-authored diabetes-management scenario (4 states, 4 treatment intensities) by value iteration, then checks the result two independent ways: an analytical Bellman-residual check, and thousands of Monte Carlo rollouts.

**The scenario data is synthetic.** Every transition probability and reward value in `backend/services/clinical_scenario.py` was hand-authored to be directionally plausible (more aggressive treatment improves control faster but carries more burden; being stuck in a bad state is costly) — none of it is derived from clinical trials, guidelines, or patient data. This demonstrates the algorithm, not medical guidance.

## How it works

1. **Value iteration** (`ClinicalMDP.value_iteration`) computes the optimal state values and policy via the Bellman optimality equation, vectorized with numpy.
2. **Bellman residual** (`ClinicalMDP.bellman_residual`) independently verifies convergence: for every state, `V(s)` must equal `max_a[R(s,a) + γ·Σ_s' T(s,a,s')V(s')]`.
3. **Monte Carlo policy evaluation** (`PolicyEvaluator.evaluate_policy`) simulates thousands of rollouts under a given policy and reports the empirical mean discounted reward — a second, independent method of computing the same quantity value iteration computes analytically.
4. **Prioritized sweeping** (`ClinicalMDP.prioritized_sweeping_value_iteration`) is an alternative solver reaching the same fixed point by repeatedly updating whichever state has the largest pending Bellman error, instead of sweeping every state each round.

## Correctness

```bash
cd backend
pytest tests/ -v
```

Real, computed result from this scenario (not illustrative — this is what the tests actually assert and what running them prints):

| Check | Result |
|---|---|
| Bellman residual after convergence | `8.3e-7` (should be ~0) |
| Value iteration vs. prioritized sweeping | Agree to `<1e-3` on every state |
| Monte Carlo (8000 rollouts) vs. exact value function | Agree within Monte Carlo sampling error |
| Optimal policy vs. fixed baseline | Optimal policy never underperforms, verified by simulation |

## The computed result

Solving the scenario (`gamma=0.9`) gives this optimal policy:

| State | Optimal action |
|---|---|
| Controlled | Lifestyle |
| Suboptimal | Dual Therapy |
| Uncontrolled | Insulin |
| Complications | Insulin |

Treatment intensity escalates as control worsens and de-escalates once controlled — this pattern emerged from solving the MDP, it was not hand-picked. Evaluated against a fixed baseline policy (always Monotherapy, regardless of state) via 2000 Monte Carlo rollouts from `Uncontrolled`:

| Policy | Mean discounted reward |
|---|---|
| Optimal (adaptive) | 58.75 ± 0.26 |
| Baseline (fixed Monotherapy) | 51.99 ± 0.36 |
| **Improvement** | **+6.77 (≈13%)** |

The Monte Carlo estimate for the optimal policy (58.75) closely matches value iteration's exact analytical value for that state (58.77) — the cross-validation `test_monte_carlo_matches_exact_value_function` checks exactly this.

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

cd backend
pytest tests/ -v
```

### Interactive demo

```bash
streamlit run app.py
```

Adjust the discount factor and starting state, solve, and see the optimal policy plus a live optimal-vs-baseline comparison.

### API

```bash
cd backend
uvicorn main:app --reload
```

or with Docker:

```bash
docker compose up --build
```

```bash
$ curl -X POST http://127.0.0.1:8000/api/solve -H "Content-Type: application/json" -d '{"gamma": 0.9}'
{"policy":{"Controlled":"Lifestyle","Suboptimal":"Dual Therapy","Uncontrolled":"Insulin","Complications":"Insulin"},
 "values":{"Controlled":78.42,"Suboptimal":69.20,"Uncontrolled":58.77,"Complications":38.28},
 "bellman_residual":8.33e-7}
```

Endpoints: `GET /health`, `GET /api/scenario`, `POST /api/solve`, `POST /api/evaluate`. Interactive docs at `/docs`.

## Project structure

```
backend/services/mdp_engine.py         Value iteration + prioritized sweeping solvers
backend/services/clinical_scenario.py  The (synthetic) scenario definition
backend/services/policy_evaluator.py   Monte Carlo policy evaluation
backend/services/simulation_service.py Orchestration: solve + evaluate
backend/main.py, routers/, schemas.py  FastAPI service
backend/tests/                         Correctness tests
app.py                                 Streamlit interactive demo
Dockerfile, docker-compose.yml         Containerized deployment
```

## Roadmap

- A partially-observable variant (POMDP) for settings where the true state isn't directly known, only inferred from noisy observations
- Additional scenarios beyond diabetes management
