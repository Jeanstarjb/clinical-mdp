from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import simulations

app = FastAPI(
    title="Clinical MDP Engine",
    description=(
        "A Markov Decision Process solver applied to an illustrative "
        "treatment-escalation scenario. Solved by value iteration and "
        "cross-checked by Monte Carlo policy evaluation -- see /api/solve "
        "and /api/evaluate."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(simulations.router)


@app.get("/health")
def health():
    return {"status": "ok"}
