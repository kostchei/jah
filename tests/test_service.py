import time

from fastapi.testclient import TestClient

from jah.engine import DecisionEngine, EngineConfig
from jah.scheduler import BoundedScheduler
from jah.service import create_app
from tests.test_engine import FakeBackend


def request_body() -> dict:
    return {
        "state": "A fact.",
        "questions": {
            "check": {
                "type": "boolean",
                "instructions": "Check the fact.",
                "proposition": "There is a fact.",
            }
        },
    }


def test_health_and_evaluate_contract() -> None:
    engine = DecisionEngine(FakeBackend(), EngineConfig(artifact_id="test-artifact"))
    with TestClient(create_app(engine=engine)) as client:
        assert client.get("/health/live").json() == {"status": "live", "artifact_id": None}
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json() == {"status": "ready", "artifact_id": "test-artifact"}

        response = client.post("/v1/evaluate", json=request_body())
        assert response.status_code == 200
        body = response.json()
        assert body["answers"]["check"] == {
            "type": "boolean",
            "value": True,
            "p_true": 0.75,
            "probabilities": {"false": 0.25, "true": 0.75},
            "calibration_status": "uncalibrated",
            "disposition": "review",
        }
        assert body["usage"]["decisions"] == 1


def test_schema_and_deadline_failures_have_documented_status_codes() -> None:
    class SlowBackend(FakeBackend):
        def score(self, question):
            time.sleep(0.05)
            return super().score(question)

    engine = DecisionEngine(SlowBackend(), EngineConfig(artifact_id="test-artifact"))
    scheduler = BoundedScheduler(maximum_concurrency=1, maximum_queue_size=0)
    with TestClient(create_app(engine=engine, scheduler=scheduler)) as client:
        malformed = client.post("/v1/evaluate", json={"state": "x", "questions": {}})
        assert malformed.status_code == 422
        expired = client.post(
            "/v1/evaluate", json=request_body(), headers={"X-JAH-Deadline-Ms": "1"}
        )
        assert expired.status_code == 504
        saturated = client.post("/v1/evaluate", json=request_body())
        assert saturated.status_code == 429


def test_failed_startup_is_live_but_not_ready() -> None:
    def fail_load():
        raise RuntimeError("missing artifact")

    with TestClient(create_app(engine_loader=fail_load)) as client:
        assert client.get("/health/live").status_code == 200
        ready = client.get("/health/ready")
        assert ready.status_code == 503
        assert ready.json()["status"] == "not_ready"
        assert client.post("/v1/evaluate", json=request_body()).status_code == 503


def test_backend_failure_is_503_and_never_returns_partial_answers() -> None:
    class FailingBackend(FakeBackend):
        def score(self, question):
            raise RuntimeError("device disappeared")

    engine = DecisionEngine(FailingBackend(), EngineConfig(artifact_id="test-artifact"))
    with TestClient(create_app(engine=engine)) as client:
        response = client.post("/v1/evaluate", json=request_body())
        assert response.status_code == 503
        assert response.json() == {"detail": "inference failed before the atomic response"}
