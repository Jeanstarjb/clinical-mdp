"""
An illustrative Type 2 diabetes management scenario, used to demonstrate
the MDP solver on something clinically-flavored.

IMPORTANT: every transition probability and reward value below is
synthetic -- hand-authored to be directionally plausible (more
aggressive treatment improves control faster but carries more burden;
being stuck in a bad state is costly), not derived from any clinical
trial, guideline, or patient data. This is a demonstration of a
decision-making algorithm, not a source of medical guidance.
"""

STATES = ["Controlled", "Suboptimal", "Uncontrolled", "Complications"]
ACTIONS = ["Lifestyle", "Monotherapy", "Dual Therapy", "Insulin"]

# transition_probs[state][action] = {next_state: probability}; each
# inner dict must sum to 1.0 (checked by ClinicalMDP.validate()).
TRANSITION_PROBS = {
    "Controlled": {
        "Lifestyle":     {"Controlled": 0.80, "Suboptimal": 0.20},
        "Monotherapy":   {"Controlled": 0.90, "Suboptimal": 0.10},
        "Dual Therapy":  {"Controlled": 0.92, "Suboptimal": 0.08},
        "Insulin":       {"Controlled": 0.85, "Suboptimal": 0.15},
    },
    "Suboptimal": {
        "Lifestyle":     {"Controlled": 0.30, "Suboptimal": 0.50, "Uncontrolled": 0.20},
        "Monotherapy":   {"Controlled": 0.50, "Suboptimal": 0.40, "Uncontrolled": 0.10},
        "Dual Therapy":  {"Controlled": 0.65, "Suboptimal": 0.30, "Uncontrolled": 0.05},
        "Insulin":       {"Controlled": 0.70, "Suboptimal": 0.25, "Uncontrolled": 0.05},
    },
    "Uncontrolled": {
        "Lifestyle":     {"Controlled": 0.05, "Suboptimal": 0.25, "Uncontrolled": 0.55, "Complications": 0.15},
        "Monotherapy":   {"Controlled": 0.15, "Suboptimal": 0.35, "Uncontrolled": 0.45, "Complications": 0.05},
        "Dual Therapy":  {"Controlled": 0.30, "Suboptimal": 0.40, "Uncontrolled": 0.28, "Complications": 0.02},
        "Insulin":       {"Controlled": 0.45, "Suboptimal": 0.35, "Uncontrolled": 0.18, "Complications": 0.02},
    },
    "Complications": {
        "Lifestyle":     {"Controlled": 0.02, "Suboptimal": 0.08, "Uncontrolled": 0.30, "Complications": 0.60},
        "Monotherapy":   {"Controlled": 0.05, "Suboptimal": 0.15, "Uncontrolled": 0.35, "Complications": 0.45},
        "Dual Therapy":  {"Controlled": 0.10, "Suboptimal": 0.20, "Uncontrolled": 0.35, "Complications": 0.35},
        "Insulin":       {"Controlled": 0.15, "Suboptimal": 0.25, "Uncontrolled": 0.35, "Complications": 0.25},
    },
}

# Per-step reward = health utility of the resulting state, minus the
# treatment burden of the action taken (side effects, cost, invasiveness).
STATE_UTILITY = {"Controlled": 10.0, "Suboptimal": 5.0, "Uncontrolled": 0.0, "Complications": -10.0}
TREATMENT_BURDEN = {"Lifestyle": 0.5, "Monotherapy": 1.5, "Dual Therapy": 3.0, "Insulin": 5.0}

REWARDS = {
    s: {a: STATE_UTILITY[s] - TREATMENT_BURDEN[a] for a in ACTIONS}
    for s in STATES
}

# A fixed "one-size-fits-all" policy representing undifferentiated
# standard care, for comparison against the MDP-optimal adaptive policy.
BASELINE_POLICY = {s: "Monotherapy" for s in STATES}

GAMMA = 0.9
