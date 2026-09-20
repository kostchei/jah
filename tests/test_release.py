from pathlib import Path

import pytest

from jah.release import build_release_manifest, verify_offline_bundle


@pytest.fixture
def repo_root():
    return Path(__file__).resolve().parents[1]


def test_build_release_manifest(repo_root, tmp_path):
    output_path = tmp_path / "release.json"
    manifest = build_release_manifest(
        "release-v0.1.0-rc1",
        model_config_path=repo_root / "configs/models/qwen3.5-4b.yaml",
        profiles_dir=repo_root / "configs/profiles/public",
        lockfile_path=repo_root / "uv.lock",
        output_path=output_path,
        rollback_release_id="release-v0.0.9",
    )
    assert manifest.release_id == "release-v0.1.0-rc1"
    assert manifest.rollback_release_id == "release-v0.0.9"
    assert manifest.manifest_sha256 is not None
    assert output_path.exists()

    # Offline verification should pass on pristine repo state
    verification = verify_offline_bundle(
        output_path,
        profiles_dir=repo_root / "configs/profiles/public",
        lockfile_path=repo_root / "uv.lock",
    )
    assert verification["valid"] is True
    assert verification["failures"] == []


def test_verify_detects_lockfile_or_profile_tampering(repo_root, tmp_path):
    output_path = tmp_path / "release.json"
    fake_lock = tmp_path / "fake.lock"
    fake_lock.write_text("lock content", encoding="utf-8")

    _ = build_release_manifest(
        "release-test",
        model_config_path=repo_root / "configs/models/qwen3.5-4b.yaml",
        profiles_dir=repo_root / "configs/profiles/public",
        lockfile_path=fake_lock,
        output_path=output_path,
    )

    # Tamper with the lockfile
    fake_lock.write_text("tampered content", encoding="utf-8")
    verification = verify_offline_bundle(
        output_path,
        profiles_dir=repo_root / "configs/profiles/public",
        lockfile_path=fake_lock,
    )
    assert verification["valid"] is False
    assert any("lockfile checksum mismatch" in f for f in verification["failures"])
