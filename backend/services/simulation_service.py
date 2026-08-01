from typing import Dict

from . import clinical_scenario
from .mdp_engine import ClinicalMDP
from .policy_evaluator import PolicyEvaluator


def build_mdp(gamma: float = None) -> ClinicalMDP:
    mdp = ClinicalMDP(
        states=clinical_scenario.STATES,
        actions=clinical_scenario.ACTIONS,
        transition_probs=clinical_scenario.TRANSITION_PROBS,
        rewards=clinical_scenario.REWARDS,
        gamma=gamma if gamma is not None else clinical_scenario.GAMMA,
    )
    mdp.validate()
    return mdp


def solve_policy(gamma: float = None) -> Dict:
    """Solve the scenario for the optimal policy via value iteration."""
    mdp = build_mdp(gamma)
    result = mdp.value_iteration()
    result['bellman_residual'] = mdp.bellman_residual(result['values'])
    return result


def evaluate_optimal_vs_baseline(start_state: str = None, n_simulations: int = 2000,
                                  gamma: float = None) -> Dict:
    """Solve for the optimal policy, then Monte Carlo-evaluate it against
    the fixed baseline ('always Monotherapy') policy under identical
    conditions -- a real, computed comparison, not an assumed one."""
    mdp = build_mdp(gamma)
    solved = mdp.value_iteration()
    optimal_policy = solved['policy']

    evaluator = PolicyEvaluator(mdp)
    comparison = evaluator.compare_policies(
        {
            'optimal': optimal_policy,
            'baseline_monotherapy': clinical_scenario.BASELINE_POLICY,
        },
        start_state=start_state,
        n_simulations=n_simulations,
    )

    optimal_mc_mean = comparison['optimal']['mean_reward']
    baseline_mc_mean = comparison['baseline_monotherapy']['mean_reward']

    return {
        'optimal_policy': optimal_policy,
        'state_values': solved['values'],
        'bellman_residual': mdp.bellman_residual(solved['values']),
        'comparison': comparison,
        'improvement_over_baseline': optimal_mc_mean - baseline_mc_mean,
    }
