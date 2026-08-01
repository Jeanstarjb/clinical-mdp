from typing import Dict, Optional

from pydantic import BaseModel, Field


class SolveRequest(BaseModel):
    gamma: float = Field(0.9, gt=0, lt=1, description="Discount factor")


class SolveResponse(BaseModel):
    policy: Dict[str, str]
    values: Dict[str, float]
    bellman_residual: float


class EvaluateRequest(BaseModel):
    start_state: Optional[str] = Field(None, description="Fix the starting state, or omit to sample uniformly")
    n_simulations: int = Field(2000, ge=100, le=20000)
    gamma: float = Field(0.9, gt=0, lt=1)


class PolicyStats(BaseModel):
    mean_reward: float
    std_error: float
    percentiles: Dict[str, float]


class EvaluateResponse(BaseModel):
    optimal_policy: Dict[str, str]
    state_values: Dict[str, float]
    bellman_residual: float
    comparison: Dict[str, PolicyStats]
    improvement_over_baseline: float


class ScenarioResponse(BaseModel):
    states: list[str]
    actions: list[str]
    transition_probs: Dict[str, Dict[str, Dict[str, float]]]
    rewards: Dict[str, Dict[str, float]]
    baseline_policy: Dict[str, str]
