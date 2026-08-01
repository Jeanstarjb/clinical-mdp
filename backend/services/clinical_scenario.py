"""
An illustrative Type 2 diabetes management scenario, used to demonstrate
the MDP solver on something clinically-flavored.

The REWARD structure below is grounded in a real, published source:

    Oh, S.-H., Lee, S.J., Noh, J., & Mo, J. (2021). Optimal treatment
    recommendations for diabetes patients using the Markov decision
    process along with the South Korean electronic health records.
    Scientific Reports, 11, 6920. https://doi.org/10.1038/s41598-021-86419-4

That paper defines a multiplicative utility-decrement reward:

    R(a, s') = RWTP * (1 - d_chronic) * (1 - d_acute) * (1 - d_risk) * (1 - d_period) - C_MED(a)

with published decrement coefficients (their Table 4):
    d_chronic = 0.105  (chronic complication present)
    d_acute   = 0.052  (acute complication present)
    d_risk    = 0.071  (Diabetes Risk Score above threshold)
    d_period  = 0.081 / 0.095 / 0.108  (diabetes duration bucket -- not
                used here, since this scenario doesn't model duration)

WHAT IS REAL vs. WHAT IS NOT, explicitly:
- d_chronic, d_acute, d_risk are real, published coefficients, used as-is.
- Their model has 72 states (5 independent clinical factors); this
  scenario has 4. Mapping our 4-state control ladder onto their three
  decrement factors (below) is *our own interpretive choice*, not
  something the paper specifies -- they never define a "Controlled /
  Suboptimal / Uncontrolled / Complications" ladder.
- RWTP (their reward-per-full-health-step baseline) is never given a
  numeric value in the paper. We use 10.0 as a normalization constant.
- C_MED(a), their medication-cost term, is also never given numeric
  values. We keep our own hand-authored relative treatment-burden
  scale for this term (rescaled to be comparable in magnitude to the
  now much narrower utility range the real decrements produce).
- TRANSITION_PROBS below remain entirely synthetic. The paper's
  transitions were learned privately from 69,446 patients' 11-year EHR
  data via occurrence counting and were never published as a table --
  no public source provides this, so it is not something we could cite.
"""

STATES = ["Controlled", "Suboptimal", "Uncontrolled", "Complications"]
ACTIONS = ["Lifestyle", "Monotherapy", "Dual Therapy", "Insulin"]

# transition_probs[state][action] = {next_state: probability}; each
# inner dict must sum to 1.0 (checked by ClinicalMDP.validate()).
# SYNTHETIC -- see module docstring.
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

# --- Reward structure: real decrement coefficients (Oh et al. 2021), our mapping ---

RWTP = 10.0  # normalization constant; paper does not publish a numeric RWTP

# Published decrement coefficients (Oh et al. 2021, Table 4).
D_CHRONIC = 0.105
D_ACUTE = 0.052
D_RISK = 0.071

# Our mapping of this scenario's 4 states onto their 3 decrement factors
# (our interpretive choice, documented above): control worsens ->
# accumulate more of the factors their model associates with disutility.
STATE_DECREMENTS = {
    "Controlled": [],
    "Suboptimal": [D_RISK],
    "Uncontrolled": [D_RISK, D_ACUTE],
    "Complications": [D_RISK, D_ACUTE, D_CHRONIC],
}


def _state_utility(decrements: list) -> float:
    multiplier = 1.0
    for d in decrements:
        multiplier *= (1 - d)
    return RWTP * multiplier


STATE_UTILITY = {s: _state_utility(STATE_DECREMENTS[s]) for s in STATES}

# C_MED(a): the paper's medication-cost term, values not published.
# Kept as our own relative treatment-burden scale (side effects, cost,
# invasiveness), rescaled to the range the real decrements now produce
# (~7.9-10.0, versus the 0-10 spread our previous invented utility used).
TREATMENT_BURDEN = {"Lifestyle": 0.05, "Monotherapy": 0.15, "Dual Therapy": 0.30, "Insulin": 0.50}

REWARDS = {
    s: {a: STATE_UTILITY[s] - TREATMENT_BURDEN[a] for a in ACTIONS}
    for s in STATES
}

# A fixed "one-size-fits-all" policy representing undifferentiated
# standard care, for comparison against the MDP-optimal adaptive policy.
BASELINE_POLICY = {s: "Monotherapy" for s in STATES}

GAMMA = 0.9
