"""Immutable release manifest, offline verification, and rollback tooling for M5."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from jah.workload import load_yaml, sha256_file, stable_json_sha256


class ReleaseManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    release_id: str
    created_at: str
    model_id: str
    revision: str
    precision: str
    prompt_version: str
    label_version: str
    dependencies_lock_sha256: str
    profiles: dict[str, str] = Field(description="Map of profile_id to profile file SHA-256")
    limits: dict[str, int] = Field(default_factory=lambda: {
        "maximum_input_tokens": 8192,
        "maximum_questions": 32,
        "maximum_options": 16,
    })
    evaluation_report_sha256: str | None = None
    rollback_release_id: str | None = None
    manifest_sha256: str | None = None


def build_release_manifest(
    release_id: str,
    *,
    model_config_path: Path | str,
    profiles_dir: Path | str,
    lockfile_path: Path | str,
    output_path: Path | str,
    evaluation_report_path: Path | str | None = None,
    rollback_release_id: str | None = None,
) -> ReleaseManifest:
    model_cfg = load_yaml(model_config_path)
    profiles_path = Path(profiles_dir)

    profiles = {}
    if profiles_path.exists():
        for file in sorted(profiles_path.glob("*.yaml")):
            profiles[file.stem] = sha256_file(file)

    eval_sha = sha256_file(evaluation_report_path) if evaluation_report_path and Path(evaluation_report_path).exists() else None

    manifest = ReleaseManifest(
        release_id=release_id,
        created_at=datetime.now(UTC).isoformat(),
        model_id=model_cfg["model_id"],
        revision=model_cfg["revision"],
        precision=model_cfg.get("precision", "bfloat16"),
        prompt_version=model_cfg.get("prompt_version", "decision-prompt-v2"),
        label_version=model_cfg.get("label_version", "latin-uppercase-bare-v2"),
        dependencies_lock_sha256=sha256_file(lockfile_path),
        profiles=profiles,
        limits={
            "maximum_input_tokens": model_cfg.get("maximum_input_tokens", 8192),
            "maximum_questions": 32,
            "maximum_options": 16,
        },
        evaluation_report_sha256=eval_sha,
        rollback_release_id=rollback_release_id,
    )

    data = manifest.model_dump(mode="json")
    data["manifest_sha256"] = stable_json_sha256({k: v for k, v in data.items() if k != "manifest_sha256"})
    final_manifest = ReleaseManifest.model_validate(data)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(final_manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return final_manifest


def verify_offline_bundle(
    manifest_path: Path | str,
    *,
    profiles_dir: Path | str,
    lockfile_path: Path | str,
) -> dict[str, Any]:
    """Verify that all components referenced by a release manifest match their pinned checksums offline."""
    m_path = Path(manifest_path)
    if not m_path.exists():
        raise FileNotFoundError(f"release manifest not found: {manifest_path}")

    raw = json.loads(m_path.read_text(encoding="utf-8"))
    manifest = ReleaseManifest.model_validate(raw)

    failures = []
    # 1. Verify lockfile checksum
    current_lock_sha = sha256_file(lockfile_path)
    if current_lock_sha != manifest.dependencies_lock_sha256:
        failures.append(f"lockfile checksum mismatch: expected {manifest.dependencies_lock_sha256}, got {current_lock_sha}")

    # 2. Verify calibration profiles
    p_dir = Path(profiles_dir)
    for profile_id, expected_sha in manifest.profiles.items():
        profile_file = p_dir / f"{profile_id}.yaml"
        if not profile_file.exists():
            failures.append(f"missing profile file: {profile_file}")
            continue
        actual_sha = sha256_file(profile_file)
        if actual_sha != expected_sha:
            failures.append(f"profile {profile_id} checksum mismatch: expected {expected_sha}, got {actual_sha}")

    # 3. Verify manifest self-hash
    computed_sha = stable_json_sha256({k: v for k, v in raw.items() if k != "manifest_sha256"})
    if computed_sha != manifest.manifest_sha256:
        failures.append(f"manifest integrity compromised: expected {manifest.manifest_sha256}, computed {computed_sha}")

    return {
        "valid": len(failures) == 0,
        "release_id": manifest.release_id,
        "failures": failures,
    }
