from __future__ import annotations

import json
import subprocess
import sys
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
    assert benchmark.derive_worker_count(32_768, 8) == 1
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


def test_generated_checkpoint_validators_reject_missing_fingerprint(
    tmp_path: Path,
) -> None:
    script = benchmark.build_remote_script(config(), workers=1, group_size=1)

    checkpoint_marker = "  python - \"${1:-$CHECKPOINT}\" <<'PY'\n"
    checkpoint_start = script.index(checkpoint_marker) + len(checkpoint_marker)
    checkpoint_validator = script[
        checkpoint_start : script.index("\nPY\n", checkpoint_start)
    ]
    valid_checkpoint = tmp_path / "valid-checkpoint.jsonl"
    valid_checkpoint.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "case_id": "a",
                        "checkpoint_fingerprint": "fp",
                    }
                ),
                json.dumps(
                    {
                        "case_id": "b",
                        "checkpoint_fingerprint": "fp",
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    valid_checkpoint_result = subprocess.run(
        [
            sys.executable,
            "-c",
            checkpoint_validator,
            str(valid_checkpoint),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert valid_checkpoint_result.returncode == 0

    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "case_id": "a",
                        "checkpoint_fingerprint": "fp",
                    }
                ),
                json.dumps({"case_id": "b"}),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_result = subprocess.run(
        [sys.executable, "-c", checkpoint_validator, str(checkpoint)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert checkpoint_result.returncode != 0

    final_marker = "python - \"$OUTPUT\" <<'PY'\n"
    final_start = script.index(final_marker) + len(final_marker)
    final_validator = script[final_start : script.index("\nPY\n", final_start)]
    output = tmp_path / "output.json"
    valid_payload = {
        "checkpoint_fingerprint": "fp",
        "matrix": {
            "planned_cases": 2,
            "all_cases_persisted": True,
        },
        "results": [
            {
                "case_id": "a",
                "checkpoint_fingerprint": "fp",
            },
            {
                "case_id": "b",
                "checkpoint_fingerprint": "fp",
            },
        ],
    }
    output.write_text(json.dumps(valid_payload), encoding="utf-8")
    valid_final_result = subprocess.run(
        [sys.executable, "-c", final_validator, str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert valid_final_result.returncode == 0

    valid_payload["results"][1] = {"case_id": "b"}
    output.write_text(json.dumps(valid_payload), encoding="utf-8")
    final_result = subprocess.run(
        [sys.executable, "-c", final_validator, str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert final_result.returncode != 0


def test_parse_resource_probe_reports_cpu_and_ram_mb() -> None:
    assert benchmark.parse_resource_probe("nproc=8\nMemTotal_kB=33554432\n") == (
        8,
        32768,
    )


def test_initial_state_contains_required_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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


def test_cleanup_recovers_before_printing_destroy_without_confirmation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    state = {"task_name": "bbox-test", "status": "training_running", "instance_id": 77}
    monkeypatch.setattr(benchmark, "load_state", lambda _: state)
    order: list[str] = []
    monkeypatch.setattr(
        benchmark,
        "recover_outputs",
        lambda *_args, **_kwargs: order.append("recover") or [],
    )
    args = SimpleNamespace(task_name="bbox-test", destroy=False, yes=False)
    assert benchmark.handle_cleanup(args) == 0
    assert order == ["recover"]
    assert "vastai destroy instance 77 --yes" in capsys.readouterr().out


def run_args(tmp_path: Path, *extra: str) -> object:
    paths = []
    for name in ("id", "id.pub", "aws.env"):
        path = tmp_path / name
        path.write_text("fixture\n", encoding="utf-8")
        paths.append(str(path))
    return benchmark.build_parser().parse_args(
        [
            "run",
            "bbox-test",
            "--run-id",
            "bbox-test-run",
            "--s3-bucket",
            "scenic-test",
            "--identity-file",
            paths[0],
            "--ssh-public-key",
            paths[1],
            "--local-secrets-env-file",
            paths[2],
            *extra,
        ]
    )


@pytest.fixture
def fake_vast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    """Fake every Vast/SSH boundary; record destroy calls; never touch a real host."""
    calls: dict[str, object] = {
        "destroyed": [],
        "destroy_returncode": 0,
        "probe": (64, 64 * 24_576 + 8_192),
        "fail_at": None,
        "exit_reports": ["running", "exit=0"],
        "unrecoverable": set(),
        "events": [],
    }

    def step(name: str, result: object = None):
        def fake(*_args: object, **_kwargs: object) -> object:
            if calls["fail_at"] == name:
                raise RuntimeError(f"{name} failed")
            return result

        return fake

    def fake_run_command(argv: list[str], **_kwargs: object):
        assert argv[:3] == ["vastai", "destroy", "instance"]
        calls["destroyed"].append(argv[3])
        calls["events"].append("destroy")
        return subprocess.CompletedProcess(
            argv, calls["destroy_returncode"], stdout="", stderr="vast refused"
        )

    def fake_ssh(_target: object, command: str, **_kwargs: object):
        if "echo present" in command:
            output = "checkpoint" if ".jsonl" in command else "final"
            presence = calls.get("remote_presence", {}).get(output, "present")
            if presence == "unreachable":
                return subprocess.CompletedProcess([], 255, stdout="", stderr="ssh: timeout")
            return subprocess.CompletedProcess([], 0, stdout=f"{presence}\n", stderr="")
        if "exit=%s" not in command:
            return step("ssh")()
        reports = calls["exit_reports"]
        report = reports.pop(0) if len(reports) > 1 else reports[0]
        calls["events"].append(f"probe:{report}")
        return subprocess.CompletedProcess([], 0, stdout=f"{report}\n", stderr="")

    def fake_copy(
        _target: object, remote: str, destination: Path, *, required: bool
    ) -> bool:
        if destination.name in calls["unrecoverable"]:
            return False
        destination.write_text(f"copied {remote}\n", encoding="utf-8")
        calls["events"].append(f"copy:{destination.name}")
        return True

    monkeypatch.setattr(benchmark, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(benchmark, "require_commands", lambda _commands: None)
    monkeypatch.setattr(benchmark, "allocate_instance", lambda _config: (17, 4242))
    monkeypatch.setattr(benchmark, "attach_ssh_key", step("attach_ssh_key"))
    monkeypatch.setattr(
        benchmark,
        "wait_for_instance_endpoint",
        step("wait_for_instance_endpoint", ("198.51.100.10", 2200)),
    )
    monkeypatch.setattr(benchmark, "wait_for_ssh", step("wait_for_ssh"))
    monkeypatch.setattr(benchmark, "copy_secrets", step("copy_secrets"))
    monkeypatch.setattr(benchmark, "ssh", fake_ssh)
    monkeypatch.setattr(benchmark, "upload_source_overlay", step("upload"))
    monkeypatch.setattr(benchmark, "bootstrap_remote_project", step("bootstrap"))
    monkeypatch.setattr(
        benchmark,
        "remote_preflight",
        lambda _target, _config: calls["probe"],
    )
    monkeypatch.setattr(benchmark, "upload_remote_script", step("upload_script"))
    monkeypatch.setattr(benchmark, "launch_remote_script", step("launch", 31337))
    monkeypatch.setattr(benchmark, "run_command", fake_run_command)
    monkeypatch.setattr(benchmark, "copy_remote_path", fake_copy)
    monkeypatch.setattr(benchmark, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(benchmark.time, "sleep", lambda _seconds: None)
    return calls


def test_run_destroy_tears_down_owned_host_when_launch_fails(
    tmp_path: Path,
    fake_vast: dict[str, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_vast["fail_at"] = "bootstrap"
    assert benchmark.handle_run(run_args(tmp_path, "--destroy")) == 1
    assert fake_vast["destroyed"] == ["4242"]
    state = benchmark.load_state("bbox-test")
    assert state["status"] == "failed_destroyed"
    assert state["error"] == "bootstrap failed"
    assert "Vast instance destroyed" in capsys.readouterr().err


def test_run_without_destroy_keeps_failed_host_for_cleanup(
    tmp_path: Path,
    fake_vast: dict[str, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_vast["fail_at"] = "bootstrap"
    assert benchmark.handle_run(run_args(tmp_path)) == 1
    assert fake_vast["destroyed"] == []
    assert benchmark.load_state("bbox-test")["status"] == "failed_kept"
    assert "cleanup bbox-test --destroy --yes" in capsys.readouterr().err


def test_run_destroy_failure_keeps_state_with_destroy_error(
    tmp_path: Path, fake_vast: dict[str, object]
) -> None:
    fake_vast["fail_at"] = "wait_for_ssh"
    fake_vast["destroy_returncode"] = 1
    assert benchmark.handle_run(run_args(tmp_path, "--destroy")) == 1
    state = benchmark.load_state("bbox-test")
    assert state["status"] == "failed_kept"
    assert state["error"] == "wait_for_ssh failed"
    assert state["destroy_error"] == "vast refused"


def test_run_without_destroy_leaves_detached_benchmark_running(
    tmp_path: Path, fake_vast: dict[str, object]
) -> None:
    assert benchmark.handle_run(run_args(tmp_path)) == 0
    assert fake_vast["events"] == []
    assert benchmark.load_state("bbox-test")["status"] == "training_running"


def test_run_destroy_waits_recovers_then_destroys_on_success(
    tmp_path: Path, fake_vast: dict[str, object]
) -> None:
    assert benchmark.handle_run(run_args(tmp_path, "--destroy")) == 0
    events = fake_vast["events"]
    assert events[:2] == ["probe:running", "probe:exit=0"]
    assert events[-1] == "destroy"
    assert {"copy:checkpoint.jsonl", "copy:final.json"} <= set(events[2:-1])
    assert fake_vast["destroyed"] == ["4242"]
    state = benchmark.load_state("bbox-test")
    assert state["status"] == "destroyed"
    assert state["remote_exit_status"] == 0
    assert (tmp_path / "artifacts" / "bbox-test" / "final.json").is_file()


def test_run_destroy_keeps_host_when_remote_benchmark_fails(
    tmp_path: Path, fake_vast: dict[str, object]
) -> None:
    fake_vast["exit_reports"] = ["exit=3"]
    assert benchmark.handle_run(run_args(tmp_path, "--destroy")) == 1
    assert fake_vast["destroyed"] == []
    state = benchmark.load_state("bbox-test")
    assert state["status"] == "failed_kept"
    assert state["remote_exit_status"] == 3
    assert "status 3" in state["error"]


def test_run_destroy_keeps_running_host_on_completion_timeout(
    tmp_path: Path, fake_vast: dict[str, object]
) -> None:
    fake_vast["exit_reports"] = ["running"]
    args = run_args(tmp_path, "--destroy", "--timeout-seconds", "0")
    assert benchmark.handle_run(args) == 1
    assert fake_vast["destroyed"] == []
    state = benchmark.load_state("bbox-test")
    assert state["status"] == "training_running"
    assert "unresolved" in state["wait_error"]


def test_run_destroy_fails_closed_when_benchmark_vanishes(
    tmp_path: Path, fake_vast: dict[str, object]
) -> None:
    fake_vast["exit_reports"] = ["vanished"]
    assert benchmark.handle_run(run_args(tmp_path, "--destroy")) == 1
    assert fake_vast["destroyed"] == []
    assert benchmark.load_state("bbox-test")["status"] == "failed_kept"


def test_run_destroy_keeps_host_when_required_outputs_unrecovered(
    tmp_path: Path, fake_vast: dict[str, object]
) -> None:
    fake_vast["unrecoverable"] = {"final.json"}
    assert benchmark.handle_run(run_args(tmp_path, "--destroy")) == 1
    assert fake_vast["destroyed"] == []
    state = benchmark.load_state("bbox-test")
    assert state["status"] == "failed_kept"
    assert "final.json" in state["error"]


def test_remote_script_records_exit_status_marker_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(benchmark, "DEFAULT_REMOTE_RUN_ROOT", str(tmp_path / "run"))
    cfg = config(
        remote_repo_dir=str(tmp_path / "missing-repo"),
        remote_env_file=str(tmp_path / "missing.env"),
    )
    script = tmp_path / "run.sh"
    script.write_text(
        benchmark.build_remote_script(cfg, workers=1, group_size=1), encoding="utf-8"
    )
    result = subprocess.run(
        ["bash", str(script)], check=False, capture_output=True, text=True
    )
    marker = Path(benchmark.remote_paths(cfg)["exit_status"])
    assert result.returncode != 0
    assert int(marker.read_text(encoding="utf-8")) == result.returncode


def test_run_honors_configured_worker_ceiling_above_default(
    tmp_path: Path, fake_vast: dict[str, object]
) -> None:
    assert benchmark.handle_run(run_args(tmp_path, "--max-workers", "48")) == 0
    state = benchmark.load_state("bbox-test")
    assert state["workers"] == 48
    assert state["group_size"] == 48


def test_run_caps_workers_at_configured_ceiling(
    tmp_path: Path, fake_vast: dict[str, object]
) -> None:
    assert benchmark.handle_run(run_args(tmp_path, "--max-workers", "4")) == 0
    assert benchmark.load_state("bbox-test")["workers"] == 4


def test_final_worker_plan_rejects_group_below_derived_workers(
    tmp_path: Path, fake_vast: dict[str, object]
) -> None:
    args = run_args(tmp_path, "--max-workers", "8", "--group-size", "4")
    assert benchmark.handle_run(args) == 1
    state = benchmark.load_state("bbox-test")
    assert state["status"] == "failed_kept"
    assert "group-size must be at least workers" in state["error"]
    assert fake_vast["destroyed"] == []


def test_run_destroy_keeps_host_when_launch_ack_is_lost(
    tmp_path: Path, fake_vast: dict[str, object]
) -> None:
    fake_vast["fail_at"] = "launch"
    assert benchmark.handle_run(run_args(tmp_path, "--destroy")) == 1
    assert fake_vast["destroyed"] == []
    state = benchmark.load_state("bbox-test")
    assert state["status"] == "failed_kept"
    assert state["launch_attempted"] is True


def test_run_destroy_keeps_host_when_post_launch_state_write_fails(
    tmp_path: Path, fake_vast: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    real_write = benchmark.write_state

    def flaky_write(state: dict[str, object]) -> None:
        if state.get("status") == "training_running":
            raise OSError("disk full")
        real_write(state)

    monkeypatch.setattr(benchmark, "write_state", flaky_write)
    assert benchmark.handle_run(run_args(tmp_path, "--destroy")) == 1
    assert fake_vast["destroyed"] == []
    assert benchmark.load_state("bbox-test")["status"] == "failed_kept"


def cleanup_args() -> SimpleNamespace:
    return SimpleNamespace(task_name="bbox-test", destroy=True, yes=True)


def test_cleanup_refuses_to_destroy_running_benchmark(
    tmp_path: Path, fake_vast: dict[str, object]
) -> None:
    assert benchmark.handle_run(run_args(tmp_path)) == 0
    fake_vast["exit_reports"] = ["running"]
    assert benchmark.handle_cleanup(cleanup_args()) == 1
    assert fake_vast["destroyed"] == []
    assert benchmark.load_state("bbox-test")["status"] == "training_running"


def test_cleanup_refuses_when_launch_outcome_unknown(
    tmp_path: Path, fake_vast: dict[str, object]
) -> None:
    fake_vast["fail_at"] = "launch"
    assert benchmark.handle_run(run_args(tmp_path)) == 1
    fake_vast["exit_reports"] = ["unknown"]
    assert benchmark.handle_cleanup(cleanup_args()) == 1
    assert fake_vast["destroyed"] == []


def test_cleanup_refuses_when_completed_outputs_missing(
    tmp_path: Path, fake_vast: dict[str, object]
) -> None:
    assert benchmark.handle_run(run_args(tmp_path)) == 0
    fake_vast["exit_reports"] = ["exit=0"]
    fake_vast["unrecoverable"] = {"final.json"}
    assert benchmark.handle_cleanup(cleanup_args()) == 1
    assert fake_vast["destroyed"] == []
    state = benchmark.load_state("bbox-test")
    assert state["status"] == "failed_kept"
    assert "final.json" in state["error"]


def test_cleanup_refuses_when_recovery_raises(
    tmp_path: Path, fake_vast: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert benchmark.handle_run(run_args(tmp_path)) == 0
    fake_vast["exit_reports"] = ["exit=0"]

    def broken_copy(*_args: object, **_kwargs: object) -> bool:
        raise RuntimeError("scp exploded")

    monkeypatch.setattr(benchmark, "copy_remote_path", broken_copy)
    assert benchmark.handle_cleanup(cleanup_args()) == 1
    assert fake_vast["destroyed"] == []


def test_cleanup_destroys_completed_benchmark_after_recovery(
    tmp_path: Path, fake_vast: dict[str, object]
) -> None:
    assert benchmark.handle_run(run_args(tmp_path)) == 0
    fake_vast["exit_reports"] = ["exit=0"]
    assert benchmark.handle_cleanup(cleanup_args()) == 0
    assert fake_vast["events"][-1] == "destroy"
    assert benchmark.load_state("bbox-test")["status"] == "destroyed"
    assert (tmp_path / "artifacts" / "bbox-test" / "final.json").is_file()


def test_cleanup_destroys_failed_benchmark_once_existing_outputs_recovered(
    tmp_path: Path, fake_vast: dict[str, object]
) -> None:
    assert benchmark.handle_run(run_args(tmp_path)) == 0
    fake_vast["exit_reports"] = ["exit=2"]
    fake_vast["remote_presence"] = {"final": "absent"}
    fake_vast["unrecoverable"] = {"final.json"}
    assert benchmark.handle_cleanup(cleanup_args()) == 0
    assert benchmark.load_state("bbox-test")["status"] == "failed_destroyed"


def test_cleanup_refuses_failed_benchmark_when_existing_final_unrecovered(
    tmp_path: Path, fake_vast: dict[str, object]
) -> None:
    assert benchmark.handle_run(run_args(tmp_path)) == 0
    fake_vast["exit_reports"] = ["exit=1"]  # e.g. final S3 upload failed
    fake_vast["unrecoverable"] = {"final.json"}
    assert benchmark.handle_cleanup(cleanup_args()) == 1
    assert fake_vast["destroyed"] == []
    assert "final.json" in benchmark.load_state("bbox-test")["error"]


def test_cleanup_destroys_never_launched_host_without_outputs(
    tmp_path: Path, fake_vast: dict[str, object]
) -> None:
    fake_vast["fail_at"] = "bootstrap"
    assert benchmark.handle_run(run_args(tmp_path)) == 1
    fake_vast["unrecoverable"] = {"checkpoint.jsonl", "benchmark.log", "final.json"}
    assert benchmark.handle_cleanup(cleanup_args()) == 0
    assert fake_vast["destroyed"] == ["4242"]
    assert benchmark.load_state("bbox-test")["status"] == "destroyed"


def test_cleanup_refuses_failed_benchmark_when_existing_checkpoint_unrecovered(
    tmp_path: Path, fake_vast: dict[str, object]
) -> None:
    assert benchmark.handle_run(run_args(tmp_path)) == 0
    fake_vast["exit_reports"] = ["exit=2"]
    fake_vast["unrecoverable"] = {"checkpoint.jsonl", "final.json"}
    assert benchmark.handle_cleanup(cleanup_args()) == 1
    assert fake_vast["destroyed"] == []
    assert "checkpoint.jsonl" in benchmark.load_state("bbox-test")["error"]


def test_cleanup_refuses_failed_benchmark_when_checkpoint_presence_unknown(
    tmp_path: Path, fake_vast: dict[str, object]
) -> None:
    assert benchmark.handle_run(run_args(tmp_path)) == 0
    fake_vast["exit_reports"] = ["exit=2"]
    fake_vast["remote_presence"] = {"checkpoint": "unreachable"}
    assert benchmark.handle_cleanup(cleanup_args()) == 1
    assert fake_vast["destroyed"] == []


def test_cleanup_destroys_early_failure_without_checkpoint_once_log_recovered(
    tmp_path: Path, fake_vast: dict[str, object]
) -> None:
    assert benchmark.handle_run(run_args(tmp_path)) == 0
    fake_vast["exit_reports"] = ["exit=21"]
    fake_vast["remote_presence"] = {"checkpoint": "absent", "final": "absent"}
    fake_vast["unrecoverable"] = {"checkpoint.jsonl", "final.json"}
    assert benchmark.handle_cleanup(cleanup_args()) == 0
    assert benchmark.load_state("bbox-test")["status"] == "failed_destroyed"
