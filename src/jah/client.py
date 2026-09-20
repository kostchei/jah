"""Small typed synchronous client for the local HTTP service."""

from __future__ import annotations

from typing import Self

import httpx

from jah.schemas import EvaluateRequest, EvaluateResponse, HealthResponse


class JahClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", *, timeout: float = 30.0) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def __enter__(self) -> Self:
        self._client.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self._client.__exit__(*args)

    def evaluate(self, request: EvaluateRequest) -> EvaluateResponse:
        response = self._client.post("/v1/evaluate", json=request.model_dump(mode="json"))
        response.raise_for_status()
        return EvaluateResponse.model_validate(response.json())

    def live(self) -> HealthResponse:
        response = self._client.get("/health/live")
        response.raise_for_status()
        return HealthResponse.model_validate(response.json())

    def ready(self) -> HealthResponse:
        response = self._client.get("/health/ready")
        response.raise_for_status()
        return HealthResponse.model_validate(response.json())
