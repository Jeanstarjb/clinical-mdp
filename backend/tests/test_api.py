from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_scenario():
    response = client.get("/api/scenario")
    assert response.status_code == 200
    data = response.json()
    assert set(data["states"]) == {"Controlled", "Suboptimal", "Uncontrolled", "Complications"}
    assert "Insulin" in data["actions"]


def test_solve():
    response = client.post("/api/solve", json={"gamma": 0.9})
    assert response.status_code == 200
    data = response.json()
    assert set(data["policy"].keys()) == {"Controlled", "Suboptimal", "Uncontrolled", "Complications"}
    assert data["bellman_residual"] < 1e-4


def test_evaluate():
    response = client.post(
        "/api/evaluate",
        json={"start_state": "Uncontrolled", "n_simulations": 500, "gamma": 0.9},
    )
    assert response.status_code == 200
    data = response.json()
    assert "optimal" in data["comparison"]
    assert "baseline_monotherapy" in data["comparison"]
    assert isinstance(data["improvement_over_baseline"], float)


def test_solve_rejects_invalid_gamma():
    response = client.post("/api/solve", json={"gamma": 1.5})
    assert response.status_code == 422
