import numpy as np

from services.mdp_engine import ClinicalMDP
from services.policy_evaluator import PolicyEvaluator
from services import clinical_scenario


def test_monte_carlo_matches_exact_value_function():
    """The strongest correctness check: value_iteration computes V(s) by
    exact dynamic programming; PolicyEvaluator estimates the same
    quantity by simulating thousands of random rollouts under the
    resulting policy. These are two independent computations of the
    same number -- if they don't agree (within Monte Carlo noise),
    something is wrong."""
    mdp = ClinicalMDP(
        states=clinical_scenario.STATES,
        actions=clinical_scenario.ACTIONS,
        transition_probs=clinical_scenario.TRANSITION_PROBS,
        rewards=clinical_scenario.REWARDS,
        gamma=clinical_scenario.GAMMA,
    )
    solved = mdp.value_iteration(epsilon=1e-10, max_iter=5000)

    evaluator = PolicyEvaluator(mdp)
    start_state = "Uncontrolled"
    mc_result = evaluator.evaluate_policy(
        solved['policy'], start_state=start_state, n_simulations=8000, max_steps=200
    )

    exact_value = solved['values'][start_state]
    mc_mean = mc_result['mean_reward']
    mc_se = mc_result['std_error']

    # Within ~5 standard errors -- generous given max_steps truncates the
    # infinite-horizon sum, but tight enough to catch a real bug.
    assert abs(exact_value - mc_mean) < 5 * mc_se + 0.5, (
        f"exact V({start_state})={exact_value:.3f} vs Monte Carlo "
        f"{mc_mean:.3f} +/- {mc_se:.3f} -- too far apart to be sampling noise"
    )


def test_optimal_policy_beats_fixed_baseline():
    """The optimal (value-iteration) policy should never do worse, in
    expectation, than a fixed baseline policy -- that's what "optimal"
    means. This is checked with real simulation, not assumed."""
    mdp = ClinicalMDP(
        states=clinical_scenario.STATES,
        actions=clinical_scenario.ACTIONS,
        transition_probs=clinical_scenario.TRANSITION_PROBS,
        rewards=clinical_scenario.REWARDS,
        gamma=clinical_scenario.GAMMA,
    )
    solved = mdp.value_iteration()
    evaluator = PolicyEvaluator(mdp)

    comparison = evaluator.compare_policies(
        {'optimal': solved['policy'], 'baseline': clinical_scenario.BASELINE_POLICY},
        n_simulations=4000,
    )

    optimal_mean = comparison['optimal']['mean_reward']
    baseline_mean = comparison['baseline']['mean_reward']
    combined_se = comparison['optimal']['std_error'] + comparison['baseline']['std_error']

    assert optimal_mean >= baseline_mean - 3 * combined_se, (
        f"optimal policy ({optimal_mean:.3f}) underperformed baseline "
        f"({baseline_mean:.3f}) by more than sampling noise explains"
    )
