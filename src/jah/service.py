"""FastAPI application for the local decision service."""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from jah.engine import DecisionEngine, EngineConfig
from jah.errors import DeadlineExceededError, InferenceUnavailableError, QueueSaturatedError
from jah.scheduler import BoundedScheduler
from jah.schemas import EvaluateRequest, EvaluateResponse, HealthResponse, Timing
from jah.workload import load_yaml

DEFAULT_DEADLINE_MS = 30_000
MAXIMUM_DEADLINE_MS = 300_000


def load_reference_engine(
    model_config_path: str | Path | None = None,
    profiles_dir: str | Path | None = None,
) -> DecisionEngine:
    """Load, validate, and warm the pinned reference artifact."""
    from jah.backends.huggingface import HuggingFaceDirectLogitBackend
    from jah.calibration import load_profile_registry

    path = Path(model_config_path or os.environ.get("JAH_MODEL_CONFIG", "configs/models/qwen3.5-4b.yaml"))
    config = load_yaml(path)
    backend = HuggingFaceDirectLogitBackend(
        config["model_id"], config["revision"], device=config["device"]
    )
    # Step 2A / ADR-07: Enforce FP32 accumulation on decision path and assert at startup
    if config.get("device") == "cuda" or (
        config.get("device") is None and getattr(backend, "device", None) and backend.device.type == "cuda"
    ):
        import torch

        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, (
            "allow_bf16_reduced_precision_reduction must be False on decision path (ADR-07)"
        )
        assert not torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction, (
            "allow_fp16_reduced_precision_reduction must be False on decision path (ADR-07)"
        )

    p_dir = Path(profiles_dir or os.environ.get("JAH_PROFILES_DIR", "configs/profiles/public"))
    profiles = load_profile_registry(p_dir)
    # Step 3A / ADR-08: Artifact discipline - refuse startup on backbone/profile mismatch
    for p_id, profile in profiles.items():
        if profile.model_id != config["model_id"] or profile.revision != config["revision"]:
            raise RuntimeError(
                f"Startup refused per ADR-08 / Step 3A: profile {p_id!r} model {profile.model_id}@{profile.revision} "
                f"does not match active serving backbone {config['model_id']}@{config['revision']}"
            )
        if profile.precision != config.get("precision", "bfloat16"):
            raise RuntimeError(
                f"Startup refused per ADR-08 / Step 3A: profile {p_id!r} precision {profile.precision} "
                f"does not match active serving precision {config.get('precision')}"
            )
    model_metadata = {
        "model_id": config.get("model_id"),
        "revision": config.get("revision"),
        "precision": config.get("precision"),
        "prompt_version": config.get("prompt_version"),
        "label_version": config.get("label_version"),
    }
    microbatch_size = int(os.environ.get("JAH_MICROBATCH_SIZE", "16"))
    enable_prefix_cache = os.environ.get("JAH_ENABLE_PREFIX_CACHE", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    force_sequential = os.environ.get("JAH_FORCE_SEQUENTIAL", "false").lower() in (
        "1",
        "true",
        "yes",
    )
    engine = DecisionEngine(
        backend,
        EngineConfig(
            artifact_id=config["artifact_id"],
            maximum_input_tokens=config["maximum_input_tokens"],
            profiles=profiles,
            model_metadata=model_metadata,
            microbatch_size=microbatch_size,
            enable_prefix_cache=enable_prefix_cache,
            force_sequential=force_sequential,
        ),
    )
    warmup = EvaluateRequest.model_validate(
        {
            "state": "Service startup validation.",
            "questions": {
                "warmup": {
                    "type": "boolean",
                    "instructions": "Return whether the state says startup validation.",
                    "proposition": "The state says startup validation.",
                }
            },
        }
    )
    engine.evaluate(warmup, request_id="req_startup_warmup")
    return engine


def create_app(
    *,
    engine: DecisionEngine | None = None,
    engine_loader: Callable[[], DecisionEngine] | None = None,
    scheduler: BoundedScheduler | None = None,
) -> FastAPI:
    """Create an app; injected engines keep contract tests GPU-independent."""

    loader = engine_loader or load_reference_engine
    request_scheduler = scheduler or BoundedScheduler()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine
        app.state.load_error = None
        if app.state.engine is None:
            try:
                app.state.engine = await asyncio.to_thread(loader)
            except Exception as exc:  # noqa: BLE001 - readiness contains startup failures
                app.state.load_error = exc
        yield

    application = FastAPI(
        title="jev-at-home",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/health/live", response_model=HealthResponse)
    async def live() -> HealthResponse:
        return HealthResponse(status="live")

    @application.get(
        "/health/ready",
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse}},
    )
    async def ready(request: Request):
        active_engine = request.app.state.engine
        if active_engine is None:
            return JSONResponse(
                status_code=503,
                content=HealthResponse(status="not_ready").model_dump(mode="json"),
            )
        return HealthResponse(status="ready", artifact_id=active_engine.config.artifact_id)

    @application.post(
        "/v1/evaluate",
        response_model=EvaluateResponse,
        responses={429: {}, 503: {}, 504: {}},
    )
    async def evaluate(
        payload: EvaluateRequest,
        request: Request,
        x_jah_deadline_ms: int = Header(
            DEFAULT_DEADLINE_MS,
            alias="X-JAH-Deadline-Ms",
            gt=0,
            le=MAXIMUM_DEADLINE_MS,
        ),
    ) -> EvaluateResponse:
        active_engine = request.app.state.engine
        if active_engine is None:
            raise HTTPException(status_code=503, detail="inference artifact is not ready")

        request_id = f"req_{uuid.uuid4().hex}"
        started = time.perf_counter()
        try:
            response, queue_ms = await request_scheduler.run(
                lambda: active_engine.evaluate(payload, request_id=request_id),
                timeout_seconds=x_jah_deadline_ms / 1_000,
            )
        except QueueSaturatedError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except DeadlineExceededError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except InferenceUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        total_ms = (time.perf_counter() - started) * 1_000
        return response.model_copy(
            update={
                "timing_ms": Timing(
                    queue=queue_ms,
                    inference=response.timing_ms.inference,
                    total=total_ms,
                )
            }
        )

    return application


app = create_app()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-config", default="configs/models/qwen3.5-4b.yaml")
    parser.add_argument("--profiles-dir", default="configs/profiles/public")
    parser.add_argument("--microbatch-size", type=int, default=16)
    parser.add_argument("--disable-prefix-cache", action="store_true", default=False)
    parser.add_argument(
        "--force-sequential",
        action="store_true",
        default=False,
        help=(
            "Serve every question through the single-item reference path (ADR-02). "
            "Used to measure the unoptimized baseline the speedup claim divides by."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.environ["JAH_MODEL_CONFIG"] = args.model_config
    os.environ["JAH_PROFILES_DIR"] = args.profiles_dir
    os.environ["JAH_MICROBATCH_SIZE"] = str(args.microbatch_size)
    os.environ["JAH_ENABLE_PREFIX_CACHE"] = "false" if args.disable_prefix_cache else "true"
    os.environ["JAH_FORCE_SEQUENTIAL"] = "true" if args.force_sequential else "false"
    import uvicorn

    uvicorn.run("jah.service:app", host=args.host, port=args.port, factory=False)


if __name__ == "__main__":
    main()
