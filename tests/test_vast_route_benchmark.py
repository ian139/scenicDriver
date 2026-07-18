from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.remote.vast_route_benchmark as benchmark


def config(**overrides: object) -> benchmark.VastRouteConfig:
    values: dict[str, object] = {
        "task_name": "bbox-test",
        "run_id": "bbox-test-run",
        "s3_bucket": "scenic-test",
        "s3_prefix": "outputs/vast/bbox-test-run",
        "local_secrets_env_file": "/tmp/aws.env",
    }
    values.update(overrides)
    return benchmark.VastRouteConfig(**values)


def test_worker_derivation_is_cpu_and_memory_bounded() -> None:
    assert benchmark.derive_worker_count(65_536, 16) == 2
    assert benchmark.derive_workers(32_768, 8) == 1
    with pytest.raises(ValueError, match="exceeds remote CPU"):
        benchmark.derive_worker_count(32_768, 2, explicit_workers=3)
    with pytest.raises(ValueError, match="insufficient"):
        benchmark.derive_worker_count(2_048, 8)


def test_offer_and_worker_config_validation() -> None:
    benchmark.validate_offer_config(benchmark.DEFAULT_OFFER_QUERY)
    benchmark.validate_offer_config("ignored", 42)
    with pytest.raises(ValueError):
        benchmark.validate_offer_config("", None)
    with pytest.raises(ValueError):
        benchmark.validate_offer_config("ignored", 0)
    with pytest.raises(ValueError, match="group-size"):
        benchmark.validate_worker_overrides(4, 2)

def test_allocation_retries_distinct_offers(monkeypatch: pytest.MonkeyPatch) -> None:
    selected: list[int] = []
    created: list[int] = []

    def select(_query: str, index: int) -> int:
        selected.append(index)
        return 100 + index

    def create(offer_id: int, _image: str, _disk_gb: int) -> int:
        created.append(offer_id)
        if len(created) < 3:
            raise RuntimeError("offer unavailable")
        return 900

    monkeypatch.setattr(benchmark, "select_offer_id_at", select)
    monkeypatch.setattr(benchmark, "create_instance", create)
    assert benchmark.allocate_instance(config(allocation_attempts=3)) == (102, 900)
    assert selected == [0, 1, 2]
    assert created == [100, 101, 102]


def test_remote_script_has_checkpoint_resume_and_final_guard() -> None:
    secret = "AWS_SECRET_ACCESS_KEY=must-never-be-embedded"
    cfg = config(remote_env_file="/root/.scenic/aws.env")
    script = benchmark.build_remote_script(cfg, workers=3, group_size=6)
    assert secret not in script
    assert "production_benchmark.py" in script
    assert "--workers 3" in script
    assert "--group-size 6" in script
    assert "--resume" in script
    assert "aws s3 cp \"$CHECKPOINT\"" in script or "aws s3 cp \"$tmp\"" in script
    assert "matrix.all_cases_persisted" in script
    assert "refusing final upload" in script
    assert "s3://scenic-test/outputs/vast/bbox-test-run/checkpoints/bbox-test/bbox-test.jsonl" in script
    assert "s3://scenic-test/outputs/vast/bbox-test-run/bbox-test.json" in script


def test_remote_script_propagates_strict_service_mode() -> None:
    script = benchmark.build_remote_script(config(strict_service_full=True), workers=1, group_size=1)
    assert "--strict-service-full" in script

def test_preflight_checks_required_tools_and_resources() -> None:
    script = benchmark.build_preflight_script(config())
    assert "nproc" in script
    assert "MemTotal" in script
    assert "df -Pk" in script
    assert "command -v uv" in script
    assert "aws sts get-caller-identity" in script
    assert "check_beta_artifacts.py" in script
    assert benchmark.parse_resource_probe("nproc=8\nMemTotal_kB=33554432\n") == (8, 32768)

def test_bootstrap_uses_explicit_canonical_artifact_source(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[str] = []
    monkeypatch.setattr(benchmark, "ssh", lambda _target, command: commands.append(command))
    benchmark.bootstrap_remote_project(
        benchmark.SshTarget("198.51.100.10", 2200, "root", "/tmp/id"),
        config(),
    )
    assert commands
    assert "--s3-bucket scenicdriver-data" in commands[0]
    assert "--s3-prefix releases/routeOptimizer/75ee0431/" in commands[0]


def test_parser_exposes_lifecycle_commands_and_dry_run() -> None:
    parser = benchmark.build_parser()
    assert parser.parse_args(["run", "task", "--dry-run"]).dry_run is True
    assert parser.parse_args(["status", "task"]).command == "status"
    assert parser.parse_args(["recover", "task"]).command == "recover"
    args = parser.parse_args(["cleanup", "task", "--destroy", "--yes"])
    assert args.command == "cleanup" and args.destroy and args.yes


def test_initial_state_contains_required_groups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark, "STATE_DIR", tmp_path / "state")
    state = benchmark.build_initial_state(
        config(),
        offer_id=17,
        instance_id=99,
        workers=4,
        group_size=8,
        cpu_count=8,
        ram_mb=32_768,
        ssh_target=benchmark.SshTarget("198.51.100.10", 2200, "root", "/tmp/id"),
    )
    benchmark.write_state(state)
    loaded = benchmark.load_state("bbox-test")
    assert loaded["instance_id"] == 99
    assert loaded["ssh_host"] == "198.51.100.10"
    assert loaded["run_id"] == "bbox-test-run"
    assert loaded["checkpoint_path"].endswith("production_artifact_benchmark.jsonl")
    assert loaded["workers"] == loaded["worker"]["count"] == 4
    assert loaded["group_size"] == loaded["group"]["size"] == 8
    assert loaded["status"] == "creating"


def test_cleanup_recovers_before_printing_destroy_without_confirmation(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    state = {"task_name": "bbox-test", "status": "training_running", "instance_id": 77}
    monkeypatch.setattr(benchmark, "load_state", lambda _: state)
    order: list[str] = []
    monkeypatch.setattr(benchmark, "recover_outputs", lambda *_args, **_kwargs: order.append("recover") or [])
    args = SimpleNamespace(task_name="bbox-test", destroy=False, yes=False)
    assert benchmark.handle_cleanup(args) == 0
    assert order == ["recover"]
    assert "vastai destroy instance 77 --yes" in capsys.readouterr().out
