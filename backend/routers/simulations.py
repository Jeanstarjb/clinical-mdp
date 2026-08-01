from fastapi import APIRouter

from schemas import EvaluateRequest, EvaluateResponse, ScenarioResponse, SolveRequest, SolveResponse
from services import clinical_scenario
from services.simulation_service import evaluate_optimal_vs_baseline, solve_policy

router = APIRouter(prefix="/api", tags=["Clinical MDP"])


@router.get("/scenario", response_model=ScenarioResponse)
def get_scenario():
    return ScenarioResponse(
        states=clinical_scenario.STATES,
        actions=clinical_scenario.ACTIONS,
        transition_probs=clinical_scenario.TRANSITION_PROBS,
        rewards=clinical_scenario.REWARDS,
        baseline_policy=clinical_scenario.BASELINE_POLICY,
    )


@router.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    result = solve_policy(gamma=req.gamma)
    return SolveResponse(
        policy=result['policy'],
        values=result['values'],
        bellman_residual=result['bellman_residual'],
    )


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    result = evaluate_optimal_vs_baseline(
        start_state=req.start_state,
        n_simulations=req.n_simulations,
        gamma=req.gamma,
    )
    return EvaluateResponse(**result)
