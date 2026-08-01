import numpy as np
from typing import Dict, List
from heapq import heappush, heappop


class ClinicalMDP:
    """A finite Markov Decision Process solved by value iteration.

    States/actions/transition_probs/rewards use plain dict keys so a
    scenario can be defined in readable, clinically-flavored terms
    (see clinical_scenario.py) while the solver itself works on
    precomputed numpy tensors for speed.
    """

    def __init__(self, states: List[str], actions: List[str],
                 transition_probs: Dict[str, Dict[str, Dict[str, float]]],
                 rewards: Dict[str, Dict[str, float]], gamma: float = 0.9):
        self.states = states
        self.actions = actions
        self.transition_probs = transition_probs
        self.rewards = rewards
        self.gamma = gamma
        self.state_idx = {s: i for i, s in enumerate(states)}
        self.action_idx = {a: i for i, a in enumerate(actions)}
        self.n_states = len(states)
        self.n_actions = len(actions)

        self.T = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions))

        for s_idx, s in enumerate(states):
            for a_idx, a in enumerate(actions):
                self.R[s_idx, a_idx] = rewards[s].get(a, 0)
                for s2 in transition_probs[s][a]:
                    self.T[s_idx, a_idx, self.state_idx[s2]] = transition_probs[s][a][s2]

    def validate(self, tol: float = 1e-6) -> None:
        """Every (state, action) row of the transition table must be a
        valid probability distribution. Raises AssertionError with the
        offending state/action if not -- this is checked before any
        solve, not assumed."""
        for s in self.states:
            for a in self.actions:
                total = sum(self.transition_probs[s][a].values())
                assert abs(total - 1.0) < tol, (
                    f"transition_probs[{s!r}][{a!r}] sums to {total}, not 1.0"
                )

    def value_iteration(self, epsilon: float = 1e-6, max_iter: int = 1000) -> dict:
        V = np.zeros(self.n_states)
        policy = np.zeros(self.n_states, dtype=int)

        for _ in range(max_iter):
            V_prev = V.copy()
            Q = self.R + self.gamma * np.einsum('ijk,k->ij', self.T, V)
            V = np.max(Q, axis=1)
            policy = np.argmax(Q, axis=1)

            if np.max(np.abs(V - V_prev)) < epsilon:
                break

        return {
            'values': {s: float(V[i]) for i, s in enumerate(self.states)},
            'policy': {s: self.actions[policy[i]] for i, s in enumerate(self.states)},
        }

    def bellman_residual(self, values: Dict[str, float]) -> float:
        """Max, over all states, of |V(s) - max_a[R(s,a) + gamma * sum_s' T V]|.
        Should be ~0 for a converged value function -- this is the
        correctness check for value_iteration, independent of any
        particular run of it."""
        V = np.array([values[s] for s in self.states])
        Q = self.R + self.gamma * np.einsum('ijk,k->ij', self.T, V)
        V_star = np.max(Q, axis=1)
        return float(np.max(np.abs(V - V_star)))

    def prioritized_sweeping_value_iteration(self, epsilon: float = 1e-6, max_iter: int = 50000) -> dict:
        """Alternative solver: same fixed point as value_iteration, reached
        by repeatedly updating whichever state has the largest pending
        Bellman error instead of sweeping every state each round."""
        V = np.zeros(self.n_states)
        policy = np.zeros(self.n_states, dtype=int)
        priority_queue = []

        for s_idx in range(self.n_states):
            heappush(priority_queue, (-np.inf, s_idx))

        # NOTE: this used to `break` the whole loop the first time any one
        # popped state had a small delta, which stops the algorithm after
        # updating only a handful of states while the rest sit at their
        # initial value of 0 -- confirmed by test_value_iteration_and_
        # prioritized_sweeping_agree disagreeing with value_iteration by
        # 3x on this scenario. Correct behavior: a state that has settled
        # just doesn't get its predecessors re-queued; the algorithm as a
        # whole only stops once the queue actually drains (or max_iter of
        # individual state-updates, not "sweeps", is hit).
        for _ in range(max_iter):
            if not priority_queue:
                break

            _, s_idx = heappop(priority_queue)
            old_value = V[s_idx]

            Q = self.R[s_idx] + self.gamma * np.dot(self.T[s_idx], V)
            V[s_idx] = np.max(Q)
            policy[s_idx] = np.argmax(Q)

            delta = abs(old_value - V[s_idx])
            if delta < epsilon:
                continue

            for a_idx in range(self.n_actions):
                for pred_idx in np.where(self.T[:, a_idx, s_idx] > 0)[0]:
                    Q_pred = self.R[pred_idx] + self.gamma * np.dot(self.T[pred_idx], V)
                    best_q = np.max(Q_pred)
                    pred_priority = abs(V[pred_idx] - best_q)
                    if pred_priority > epsilon:
                        heappush(priority_queue, (-pred_priority, pred_idx))

        return {
            'values': {s: float(V[i]) for i, s in enumerate(self.states)},
            'policy': {s: self.actions[policy[i]] for i, s in enumerate(self.states)},
        }
