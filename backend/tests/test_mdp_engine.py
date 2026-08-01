import numpy as np
import pytest

from services.mdp_engine import ClinicalMDP
from services import clinical_scenario


def test_value_iteration_toy_example():
    """A 2-state MDP small enough to reason about by hand."""
    states = ['s0', 's1']
    actions = ['a0', 'a1']
    transition_probs = {
        's0': {'a0': {'s0': 0.5, 's1': 0.5}, 'a1': {'s0': 0.1, 's1': 0.9}},
        's1': {'a0': {'s0': 0.8, 's1': 0.2}, 'a1': {'s0': 0.0, 's1': 1.0}},
    }
    rewards = {'s0': {'a0': 5, 'a1': 10}, 's1': {'a0': -1, 'a1': 2}}
    mdp = ClinicalMDP(states, actions, transition_probs, rewards, gamma=0.9)
    result = mdp.value_iteration(epsilon=1e-8)

    assert set(result['values'].keys()) == set(states)
    assert result['policy']['s0'] in actions
    # a1 from s1 always self-loops with reward 2, so its value is the
    # closed-form geometric series 2 / (1 - 0.9) if that action is optimal.
    if result['policy']['s1'] == 'a1':
        assert np.isclose(result['values']['s1'], 2 / (1 - 0.9), rtol=1e-3)


def test_bellman_residual_near_zero_after_convergence():
    """The core correctness check: at a converged value function, V(s)
    must equal max_a[R(s,a) + gamma * sum T*V] for every state. This is
    checked directly, not assumed from the solver having "finished"."""
    mdp = ClinicalMDP(
        states=clinical_scenario.STATES,
        actions=clinical_scenario.ACTIONS,
        transition_probs=clinical_scenario.TRANSITION_PROBS,
        rewards=clinical_scenario.REWARDS,
        gamma=clinical_scenario.GAMMA,
    )
    mdp.validate()
    result = mdp.value_iteration(epsilon=1e-10, max_iter=5000)
    residual = mdp.bellman_residual(result['values'])
    assert residual < 1e-6, f"Bellman residual {residual} -- value function has not converged"


def test_value_iteration_and_prioritized_sweeping_agree():
    """Two independently-implemented solvers reaching the same fixed
    point is real evidence neither has a bug that happens to be
    self-consistent."""
    mdp = ClinicalMDP(
        states=clinical_scenario.STATES,
        actions=clinical_scenario.ACTIONS,
        transition_probs=clinical_scenario.TRANSITION_PROBS,
        rewards=clinical_scenario.REWARDS,
        gamma=clinical_scenario.GAMMA,
    )
    vi = mdp.value_iteration(epsilon=1e-10, max_iter=5000)
    ps = mdp.prioritized_sweeping_value_iteration(epsilon=1e-10, max_iter=20000)

    for s in mdp.states:
        assert np.isclose(vi['values'][s], ps['values'][s], atol=1e-3), (
            f"state {s}: value_iteration={vi['values'][s]} vs "
            f"prioritized_sweeping={ps['values'][s]}"
        )
        assert vi['policy'][s] == ps['policy'][s], f"state {s}: policies disagree"


def test_validate_rejects_bad_probabilities():
    states, actions = ['s0'], ['a0']
    bad_transition_probs = {'s0': {'a0': {'s0': 0.5}}}  # doesn't sum to 1
    rewards = {'s0': {'a0': 1.0}}
    mdp = ClinicalMDP(states, actions, bad_transition_probs, rewards)
    with pytest.raises(AssertionError):
        mdp.validate()
