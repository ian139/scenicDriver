from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


import pytest  # noqa: E402

from scripts.remote import cmux_vast_host  # noqa: E402


def make_up_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "task_name": "retry-smoke",
        "offer_query": "num_gpus=1",
        "offer_id": None,
        "allocation_attempts": 2,
        "image": "ian139/scenicdriver-remote-training:smoke",
        "disk_gb": 32,
        "identity_file": "~/.ssh/id_ed25519",
        "ssh_public_key": "~/.ssh/id_ed25519.pub",
        "local_secrets_env_file": ".secrets/aws.env",
        "remote_repo_dir": "/workspace/scenic-drive",
        "branch": "Ian139/RemoteTraining",
        "workspace_cwd": "/tmp/scenic-drive",
        "cmux_focus": False,
        "timeout_seconds": 600,
        "bootstrap": "none",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")




def test_ssh_commands_ignore_ephemeral_vast_known_hosts() -> None:
    target = cmux_vast_host.SshTarget(
        host="ssh3.vast.ai",
        port=12908,
        user="root",
        identity_file="/tmp/id_ed25519",
    )

    ssh_command = cmux_vast_host.ssh_base(target)
    scp_command = cmux_vast_host.scp_base(target)

    for command in (ssh_command, scp_command):
        assert "StrictHostKeyChecking=no" in command
        assert "UserKnownHostsFile=/dev/null" in command
def test_select_offer_id_at_returns_indexed_offer(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_command(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        assert command[:3] == ["vastai", "search", "offers"]
        return completed('[{"id": 101}, {"id": 202}]')

    monkeypatch.setattr(cmux_vast_host, "run_command", fake_run_command)

    assert cmux_vast_host.select_offer_id_at("num_gpus=1", 1) == 202
    with pytest.raises(SystemExit, match="No Vast offers matched query"):
        cmux_vast_host.select_offer_id_at("num_gpus=1", 2)


def test_do_up_retries_destroying_failed_ssh_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    args = make_up_args()
    created = iter([111, 222])
    endpoint_calls: list[int] = []
    ssh_attempts: list[cmux_vast_host.SshTarget] = []
    destroy_commands: list[list[str]] = []
    bootstrap_calls: list[str] = []
    state_writes: list[dict] = []

    monkeypatch.setattr(cmux_vast_host, "create_instance", lambda offer_id, image, disk_gb: next(created))
    monkeypatch.setattr(cmux_vast_host, "attach_ssh_key", lambda instance_id, public_key: None)
    monkeypatch.setattr(cmux_vast_host, "reboot_instance", lambda instance_id: None)
    monkeypatch.setattr(cmux_vast_host, "write_state", lambda state: state_writes.append(dict(state)))

    def fake_update_status(state: dict, status: str, **updates: object) -> None:
        state.update(updates)
        state["status"] = status
        state_writes.append(dict(state))


    def fake_run_command(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["vastai", "search", "offers"]:
            return completed('[{"id": 9001}, {"id": 9002}]')
        if command[:3] == ["vastai", "destroy", "instance"]:
            destroy_commands.append(command)
            return completed()
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(cmux_vast_host, "run_command", fake_run_command)

    def fake_wait_for_instance_endpoint(instance_id: int, timeout_seconds: int) -> tuple[str, int]:
        endpoint_calls.append(instance_id)
        return (f"host-{instance_id}", 22000 + instance_id)

    monkeypatch.setattr(cmux_vast_host, "wait_for_instance_endpoint", fake_wait_for_instance_endpoint)

    def fake_wait_for_ssh(target: cmux_vast_host.SshTarget, timeout_seconds: int) -> None:
        ssh_attempts.append(target)
        if len(ssh_attempts) == 1:
            raise RuntimeError("ssh failed")

    monkeypatch.setattr(cmux_vast_host, "wait_for_ssh", fake_wait_for_ssh)

    state = cmux_vast_host.do_up(args)

    assert destroy_commands == [["vastai", "destroy", "instance", "111", "--yes"]]
    assert endpoint_calls == [111, 111, 222, 222]
    assert len(ssh_attempts) == 2
    assert bootstrap_calls == []
    assert state["status"] == "ready"
    assert state["allocation_attempt"] == 2
    assert state["ssh_host"] == "host-222"
    assert state["ssh_port"] == 22222
    assert any(write.get("status") == "destroyed" and write.get("instance_id") == 111 for write in state_writes)



def test_do_up_retries_create_instance_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    args = make_up_args()
    create_offer_ids: list[int] = []
    destroy_commands: list[list[str]] = []

    monkeypatch.setattr(cmux_vast_host, "check_up_preconditions", lambda _: None)
    monkeypatch.setattr(cmux_vast_host, "write_state", lambda state: None)
    monkeypatch.setattr(
        cmux_vast_host,
        "update_status",
        lambda state, status, **updates: (state.update(updates), state.update({"status": status})),
    )
    monkeypatch.setattr(cmux_vast_host, "attach_ssh_key", lambda instance_id, public_key: None)
    monkeypatch.setattr(cmux_vast_host, "reboot_instance", lambda instance_id: None)
    monkeypatch.setattr(cmux_vast_host, "wait_for_instance_endpoint", lambda instance_id, timeout_seconds: ("host-333", 22333))
    monkeypatch.setattr(cmux_vast_host, "wait_for_ssh", lambda target, timeout_seconds: None)

    def fake_create_instance(offer_id: int, image: str, disk_gb: int) -> int:
        create_offer_ids.append(offer_id)
        if len(create_offer_ids) == 1:
            raise RuntimeError("create failed")
        return 333

    monkeypatch.setattr(cmux_vast_host, "create_instance", fake_create_instance)

    def fake_run_command(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["vastai", "search", "offers"]:
            return completed('[{"id": 9001}, {"id": 9002}]')
        if command[:3] == ["vastai", "destroy", "instance"]:
            destroy_commands.append(command)
            return completed()
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(cmux_vast_host, "run_command", fake_run_command)

    state = cmux_vast_host.do_up(args)

    assert create_offer_ids == [9001, 9002]
    assert destroy_commands == []
    assert state["status"] == "ready"
    assert state["allocation_attempt"] == 2
    assert state["instance_id"] == 333

def test_handle_start_task_returns_after_new_bootstrap_none_host(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    state = {
        "task_name": "ssh-only",
        "status": "ready",
        "identity_file": "/tmp/id_ed25519",
        "ssh_host": "203.0.113.10",
        "ssh_port": 2222,
    }
    args = argparse.Namespace(task_name="ssh-only", workspace_name=None, bootstrap="none")
    monkeypatch.setattr(cmux_vast_host, "maybe_load_state", lambda task_name: None)
    monkeypatch.setattr(cmux_vast_host, "do_up", lambda current_args: state)

    assert cmux_vast_host.handle_start_task(args) == 0
    output = capsys.readouterr().out
    assert "Vast host ready (bootstrap none): ssh-only" in output
    assert "ssh -i /tmp/id_ed25519 -p 2222" in output
    assert "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@203.0.113.10" in output

def test_legacy_state_is_read_and_normalized_to_cmux_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cmux_state_dir = tmp_path / ".cmux-vast" / "state"
    legacy_state_dir = tmp_path / ".orca-vast" / "state"
    legacy_state_dir.mkdir(parents=True)
    (legacy_state_dir / "legacy.json").write_text(
        '{"task_name":"legacy","status":"worktree_running","orca_worktree_id":"w-1",'
        '"orca_worktree_path":"/workspace/scenic-drive","orca_worktree_name":"legacy-task",'
        '"orca_agent":"none"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(cmux_vast_host, "STATE_DIR", cmux_state_dir)
    monkeypatch.setattr(cmux_vast_host, "LEGACY_STATE_DIR", legacy_state_dir)

    state = cmux_vast_host.load_state("legacy")

    assert state["cmux_workspace_id"] is None
    assert state["cmux_workspace_path"] is None
    assert state["cmux_workspace_registered"] is False
    assert state["remote_repo_dir"] == "/workspace/scenic-drive"
    assert "orca_worktree_id" not in state
    cmux_vast_host.write_state(state)
    assert (cmux_state_dir / "legacy.json").exists()
    assert not (cmux_state_dir / "legacy.json").read_text(encoding="utf-8").find('"orca_') >= 0


def test_cmux_workspace_command_uses_documented_flags_without_running_cli() -> None:
    assert cmux_vast_host.build_cmux_workspace_command("legacy-task", "/tmp/scenic-drive") == [
        "cmux",
        "new-workspace",
        "--name",
        "legacy-task",
        "--cwd",
        "/tmp/scenic-drive",
        "--focus",
        "false",
    ]

@pytest.mark.parametrize(
    ("returncode", "stdout"),
    [
        (1, ""),
        (0, "not-json"),
        (0, "{}"),
    ],
)
def test_watch_cmux_uncertainty_keeps_pending_without_teardown(
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    stdout: str,
) -> None:
    state = {
        "task_name": "cmux-uncertain",
        "status": "workspace_pending",
        "instance_id": 12345,
        "cmux_workspace_id": "workspace-1",
        "cmux_workspace_path": "/tmp/scenic-drive",
    }
    commands: list[list[str]] = []
    copy_calls: list[dict] = []
    destroy_calls: list[dict] = []

    def fake_run_command(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return completed(stdout=stdout, returncode=returncode)

    monkeypatch.setattr(cmux_vast_host, "run_command", fake_run_command)
    monkeypatch.setattr(cmux_vast_host, "copy_artifacts", lambda current: copy_calls.append(current))
    monkeypatch.setattr(cmux_vast_host, "destroy_instance", lambda current: destroy_calls.append(current))

    cmux_vast_host.process_watch_state(state, destroy=True, yes=True)

    assert commands == [["cmux", "workspace", "list", "--json"]]
    assert copy_calls == []
    assert destroy_calls == []
    assert state["status"] == "workspace_pending"


def test_watch_empty_cmux_workspace_list_without_identity_keeps_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "task_name": "cmux-no-identity",
        "status": "workspace_pending",
        "instance_id": 12345,
    }
    copy_calls: list[dict] = []
    destroy_calls: list[dict] = []

    monkeypatch.setattr(cmux_vast_host, "run_command", lambda command, **_: completed("[]"))
    monkeypatch.setattr(cmux_vast_host, "copy_artifacts", lambda current: copy_calls.append(current))
    monkeypatch.setattr(cmux_vast_host, "destroy_instance", lambda current: destroy_calls.append(current))

    cmux_vast_host.process_watch_state(state, destroy=True, yes=True)

    assert copy_calls == []
    assert destroy_calls == []
    assert state["status"] == "workspace_pending"


def test_watch_required_artifact_copy_failure_does_not_destroy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "task_name": "cmux-copy-failure",
        "status": "workspace_pending",
        "instance_id": 12345,
        "cmux_workspace_id": "workspace-1",
    }
    destroy_calls: list[dict] = []

    monkeypatch.setattr(cmux_vast_host, "workspace_closed", lambda current: True)
    monkeypatch.setattr(cmux_vast_host, "copy_artifacts", lambda current: False)
    monkeypatch.setattr(cmux_vast_host, "destroy_instance", lambda current: destroy_calls.append(current))

    cmux_vast_host.process_watch_state(state, destroy=True, yes=True)

    assert destroy_calls == []
    assert state["status"] == "workspace_pending"

def test_watch_missing_required_remote_path_does_not_destroy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "task_name": "cmux-missing-artifact",
        "status": "workspace_pending",
        "instance_id": 12345,
        "cmux_workspace_id": "workspace-1",
        "ssh_host": "ssh5.vast.ai",
        "ssh_port": 2222,
        "identity_file": "/tmp/id_ed25519",
        "remote_repo_dir": "/workspace/scenic-drive",
    }
    copy_requests: list[tuple[str, bool]] = []
    destroy_calls: list[dict] = []

    def fake_copy_remote_path(target: object, remote_path: str, local_path: Path, *, required: bool) -> bool:
        copy_requests.append((remote_path, required))
        return not required
    monkeypatch.setattr(cmux_vast_host, "workspace_closed", lambda current: True)
    monkeypatch.setattr(cmux_vast_host, "copy_remote_path", fake_copy_remote_path)
    monkeypatch.setattr(cmux_vast_host, "destroy_instance", lambda current: destroy_calls.append(current))

    cmux_vast_host.process_watch_state(state, destroy=True, yes=True)

    assert any(required for _, required in copy_requests)
    assert destroy_calls == []
    assert state["status"] == "workspace_pending"




def test_partial_migration_does_not_confirm_cmux_from_legacy_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "task_name": "cmux-partial-migration",
        "status": "workspace_pending",
        "instance_id": 12345,
        "remote_repo_dir": "/workspace/scenic-drive",
        "cmux_workspace_id": None,
        "cmux_workspace_path": None,
        "orca_worktree_id": "legacy-worktree",
        "orca_worktree_path": "/workspace/scenic-drive",
    }
    state = cmux_vast_host._normalize_state(state)
    assert state["cmux_workspace_id"] is None
    assert state["cmux_workspace_path"] is None
    assert state["remote_repo_dir"] == "/workspace/scenic-drive"

    destroy_calls: list[dict] = []
    monkeypatch.setattr(
        cmux_vast_host,
        "run_command",
        lambda command, **_: completed(
            '[{"ref":"legacy-worktree","current_directory":"/workspace/scenic-drive"}]'
        ),
    )
    monkeypatch.setattr(cmux_vast_host, "copy_artifacts", lambda current: pytest.fail("copy requires confirmed CMUX closure"))
    monkeypatch.setattr(cmux_vast_host, "destroy_instance", lambda current: destroy_calls.append(current))

    cmux_vast_host.process_watch_state(state, destroy=True, yes=True)

    assert destroy_calls == []
    assert state["status"] == "workspace_pending"

def test_watch_destroys_after_observed_workspace_disappears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "task_name": "cmux-observed",
        "status": "workspace_pending",
        "instance_id": 12345,
        "cmux_workspace_ref": "workspace-ref-1",
    }
    responses = iter(
        [
            completed('[{"ref":"workspace-ref-1"}]'),
            completed('[{"ref":"unrelated-workspace"}]'),
        ]
    )
    commands: list[list[str]] = []
    writes: list[dict] = []
    copy_calls: list[dict] = []
    destroy_calls: list[dict] = []

    def fake_run_command(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return next(responses)

    def fake_copy_artifacts(current: dict) -> bool:
        copy_calls.append(current)
        return True

    def fake_destroy_instance(current: dict) -> None:
        destroy_calls.append(current)
        current["status"] = "destroyed"

    monkeypatch.setattr(cmux_vast_host, "run_command", fake_run_command)
    monkeypatch.setattr(cmux_vast_host, "write_state", lambda current: writes.append(dict(current)))
    monkeypatch.setattr(cmux_vast_host, "copy_artifacts", fake_copy_artifacts)
    monkeypatch.setattr(cmux_vast_host, "destroy_instance", fake_destroy_instance)

    cmux_vast_host.process_watch_state(state, destroy=True, yes=True)

    assert state["status"] == "workspace_pending"
    assert state["cmux_workspace_observed"] is True
    assert state["cmux_workspace_registered"] is True
    assert state["cmux_workspace_observed_ref"] == "workspace-ref-1"
    assert copy_calls == []
    assert destroy_calls == []

    cmux_vast_host.process_watch_state(state, destroy=True, yes=True)

    assert commands == [
        ["cmux", "workspace", "list", "--json"],
        ["cmux", "workspace", "list", "--json"],
    ]
    assert len(writes) == 1
    assert copy_calls == [state]
    assert destroy_calls == [state]
    assert state["status"] == "destroyed"


def test_handle_start_task_registers_workspace_and_persists_distinct_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "task_name": "register-me",
        "status": "ready",
        "workspace_cwd": "/tmp/cmux-register",
        "cmux_workspace_registered": False,
        "cmux_workspace_ref": None,
        "cmux_workspace_id": None,
    }
    args = argparse.Namespace(
        task_name="register-me",
        workspace_name="registered-task",
        bootstrap="full",
    )
    writes: list[dict] = []
    commands: list[list[str]] = []

    monkeypatch.setattr(cmux_vast_host, "maybe_load_state", lambda task_name: state)
    monkeypatch.setattr(cmux_vast_host, "write_state", lambda current: writes.append(dict(current)))

    def fake_run_command(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert command[:3] == ["cmux", "rpc", "workspace.create"]
        assert json.loads(command[3]) == {
            "cwd": "/tmp/cmux-register",
            "focus": False,
            "name": "registered-task",
        }
        return completed(
            '{"workspace_ref":"cmux-ref-7","workspace_id":"6e2f3e0c-3a2d-4c9d-8f54-9a0e2f7d1b63"}'
        )

    monkeypatch.setattr(cmux_vast_host, "run_command", fake_run_command)

    assert cmux_vast_host.handle_start_task(args) == 0

    assert len(commands) == 1
    assert state["status"] == "workspace_pending"
    assert state["cmux_workspace_ref"] == "cmux-ref-7"
    assert state["cmux_workspace_id"] == "6e2f3e0c-3a2d-4c9d-8f54-9a0e2f7d1b63"
    assert state["cmux_workspace_registered"] is True
    assert writes[-1]["cmux_workspace_ref"] == "cmux-ref-7"
    assert writes[-1]["cmux_workspace_id"] == "6e2f3e0c-3a2d-4c9d-8f54-9a0e2f7d1b63"
    assert writes[-1]["cmux_workspace_registered"] is True


def test_handle_start_task_registration_failure_keeps_pending_without_identity_or_destroy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "task_name": "registration-retry",
        "status": "ready",
        "workspace_cwd": "/tmp/cmux-retry",
        "cmux_workspace_registered": False,
        "cmux_workspace_ref": None,
        "cmux_workspace_id": None,
    }
    args = argparse.Namespace(
        task_name="registration-retry",
        workspace_name="retry-task",
        bootstrap="full",
    )
    destroy_calls: list[dict] = []
    writes: list[dict] = []

    monkeypatch.setattr(cmux_vast_host, "maybe_load_state", lambda task_name: state)
    monkeypatch.setattr(cmux_vast_host, "write_state", lambda current: writes.append(dict(current)))
    monkeypatch.setattr(cmux_vast_host, "destroy_instance", lambda current: destroy_calls.append(current))
    monkeypatch.setattr(
        cmux_vast_host,
        "run_command",
        lambda command, **_: completed(returncode=1),
    )

    assert cmux_vast_host.handle_start_task(args) == 1

    assert state["status"] == "workspace_pending"
    assert state["cmux_workspace_ref"] is None
    assert state["cmux_workspace_id"] is None
    assert state["cmux_workspace_registered"] is False
    assert "cmux_workspace_observed_ref" not in state
    assert writes[-1]["status"] == "workspace_pending"
    assert writes[-1]["cmux_workspace_ref"] is None
    assert writes[-1]["cmux_workspace_id"] is None
    assert destroy_calls == []


def test_handle_start_task_skips_workspace_create_when_already_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "task_name": "already-registered",
        "status": "workspace_pending",
        "workspace_cwd": "/tmp/cmux-existing",
        "cmux_workspace_ref": "cmux-ref-existing",
        "cmux_workspace_id": "8b7b6d3e-cf9d-41e2-8c33-6b68f0a3b0e1",
        "cmux_workspace_registered": True,
    }
    args = argparse.Namespace(
        task_name="already-registered",
        workspace_name="existing-task",
        bootstrap="full",
    )
    commands: list[list[str]] = []

    monkeypatch.setattr(cmux_vast_host, "maybe_load_state", lambda task_name: state)
    monkeypatch.setattr(cmux_vast_host, "run_command", lambda command, **_: commands.append(command))

    assert cmux_vast_host.handle_start_task(args) == 0

    assert commands == []
    assert state["cmux_workspace_ref"] == "cmux-ref-existing"
    assert state["cmux_workspace_id"] == "8b7b6d3e-cf9d-41e2-8c33-6b68f0a3b0e1"
