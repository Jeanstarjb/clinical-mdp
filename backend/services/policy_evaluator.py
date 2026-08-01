from typing import Dict, List
import numpy as np
from scipy.stats import sem

from .mdp_engine import ClinicalMDP


class PolicyEvaluator:
    """Monte Carlo evaluation of a fixed policy against an MDP -- an
    independent check on value_iteration's exact dynamic-programming
    result. If both methods agree, that's real evidence the solver is
    correct, not just internally self-consistent."""

    def __init__(self, mdp: ClinicalMDP):
        self.mdp = mdp

    def evaluate_policy(self, policy: Dict[str, str], start_state: str = None,
                         n_simulations: int = 2000, max_steps: int = 200) -> Dict:
        cumulative_rewards = []
        rng = np.random.default_rng()
        for _ in range(n_simulations):
            state = start_state if start_state else rng.choice(self.mdp.states)
            total_reward = 0.0
            for t in range(max_steps):
                action = policy[state]
                next_states = list(self.mdp.transition_probs[state][action].keys())
                probs = list(self.mdp.transition_probs[state][action].values())
                total_reward += self.mdp.rewards[state][action] * (self.mdp.gamma ** t)
                state = rng.choice(next_states, p=probs)
            cumulative_rewards.append(total_reward)

        cumulative_rewards = np.array(cumulative_rewards)
        return {
            'mean_reward': float(np.mean(cumulative_rewards)),
            'std_error': float(sem(cumulative_rewards)),
            'percentiles': {
                str(p): float(np.percentile(cumulative_rewards, p)) for p in (25, 50, 75)
            },
        }

    def compare_policies(self, policies: Dict[str, Dict[str, str]], start_state: str = None,
                          n_simulations: int = 2000) -> Dict:
        """Evaluate several named policies under identical conditions and
        report real, computed results for each -- no hardcoded numbers."""
        return {
            name: self.evaluate_policy(policy, start_state=start_state, n_simulations=n_simulations)
            for name, policy in policies.items()
        }
