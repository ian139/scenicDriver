from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


import pytest  # noqa: E402

from scripts.remote import orca_vast_host  # noqa: E402


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
        "orca_port": 6768,
        "timeout_seconds": 600,
        "bootstrap": "none",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")




def test_ssh_commands_ignore_ephemeral_vast_known_hosts() -> None:
    target = orca_vast_host.SshTarget(
        host="ssh3.vast.ai",
        port=12908,
        user="root",
        identity_file="/tmp/id_ed25519",
    )

    ssh_command = orca_vast_host.ssh_base(target)
    scp_command = orca_vast_host.scp_base(target)

    for command in (ssh_command, scp_command):
        assert "StrictHostKeyChecking=no" in command
        assert "UserKnownHostsFile=/dev/null" in command
def test_select_offer_id_at_returns_indexed_offer(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_command(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        assert command[:3] == ["vastai", "search", "offers"]
        return completed('[{"id": 101}, {"id": 202}]')

    monkeypatch.setattr(orca_vast_host, "run_command", fake_run_command)

    assert orca_vast_host.select_offer_id_at("num_gpus=1", 1) == 202
    with pytest.raises(SystemExit, match="No Vast offers matched query"):
        orca_vast_host.select_offer_id_at("num_gpus=1", 2)


def test_do_up_retries_destroying_failed_ssh_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    args = make_up_args()
    created = iter([111, 222])
    endpoint_calls: list[int] = []
    ssh_attempts: list[orca_vast_host.SshTarget] = []
    destroy_commands: list[list[str]] = []
    bootstrap_calls: list[str] = []
    state_writes: list[dict] = []

    monkeypatch.setattr(orca_vast_host, "check_up_preconditions", lambda _: None)
    monkeypatch.setattr(orca_vast_host, "orca_status_ready", lambda: None)
    monkeypatch.setattr(orca_vast_host, "create_instance", lambda offer_id, image, disk_gb: next(created))
    monkeypatch.setattr(orca_vast_host, "attach_ssh_key", lambda instance_id, public_key: None)
    monkeypatch.setattr(orca_vast_host, "reboot_instance", lambda instance_id: None)
    monkeypatch.setattr(orca_vast_host, "write_state", lambda state: state_writes.append(dict(state)))

    def fake_update_status(state: dict, status: str, **updates: object) -> None:
        state.update(updates)
        state["status"] = status
        state_writes.append(dict(state))

    monkeypatch.setattr(orca_vast_host, "update_status", fake_update_status)
    monkeypatch.setattr(orca_vast_host, "stop_tunnel", lambda state: None)

    def fake_run_command(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["vastai", "search", "offers"]:
            return completed('[{"id": 9001}, {"id": 9002}]')
        if command[:3] == ["vastai", "destroy", "instance"]:
            destroy_commands.append(command)
            return completed()
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(orca_vast_host, "run_command", fake_run_command)

    def fake_wait_for_instance_endpoint(instance_id: int, timeout_seconds: int) -> tuple[str, int]:
        endpoint_calls.append(instance_id)
        return (f"host-{instance_id}", 22000 + instance_id)

    monkeypatch.setattr(orca_vast_host, "wait_for_instance_endpoint", fake_wait_for_instance_endpoint)

    def fake_wait_for_ssh(target: orca_vast_host.SshTarget, timeout_seconds: int) -> None:
        ssh_attempts.append(target)
        if len(ssh_attempts) == 1:
            raise RuntimeError("ssh failed")

    monkeypatch.setattr(orca_vast_host, "wait_for_ssh", fake_wait_for_ssh)

    for name in (
        "copy_secrets",
        "upload_repo",
        "bootstrap_remote",
        "run_remote_container_smoke",
        "verify_remote_smoke",
        "install_orca_server",
        "start_remote_orca_server",
        "start_tunnel",
        "wait_for_pairing_url",
        "orca_environment_add",
    ):
        monkeypatch.setattr(orca_vast_host, name, lambda *_, _name=name, **__: bootstrap_calls.append(_name))

    state = orca_vast_host.do_up(args)

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

    monkeypatch.setattr(orca_vast_host, "check_up_preconditions", lambda _: None)
    monkeypatch.setattr(orca_vast_host, "orca_status_ready", lambda: None)
    monkeypatch.setattr(orca_vast_host, "write_state", lambda state: None)
    monkeypatch.setattr(
        orca_vast_host,
        "update_status",
        lambda state, status, **updates: (state.update(updates), state.update({"status": status})),
    )
    monkeypatch.setattr(orca_vast_host, "attach_ssh_key", lambda instance_id, public_key: None)
    monkeypatch.setattr(orca_vast_host, "reboot_instance", lambda instance_id: None)
    monkeypatch.setattr(orca_vast_host, "wait_for_instance_endpoint", lambda instance_id, timeout_seconds: ("host-333", 22333))
    monkeypatch.setattr(orca_vast_host, "wait_for_ssh", lambda target, timeout_seconds: None)

    def fake_create_instance(offer_id: int, image: str, disk_gb: int) -> int:
        create_offer_ids.append(offer_id)
        if len(create_offer_ids) == 1:
            raise RuntimeError("create failed")
        return 333

    monkeypatch.setattr(orca_vast_host, "create_instance", fake_create_instance)

    def fake_run_command(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["vastai", "search", "offers"]:
            return completed('[{"id": 9001}, {"id": 9002}]')
        if command[:3] == ["vastai", "destroy", "instance"]:
            destroy_commands.append(command)
            return completed()
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(orca_vast_host, "run_command", fake_run_command)

    state = orca_vast_host.do_up(args)

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
    args = argparse.Namespace(task_name="ssh-only", worktree_name=None, bootstrap="none")

    monkeypatch.setattr(orca_vast_host, "maybe_load_state", lambda task_name: None)
    monkeypatch.setattr(orca_vast_host, "do_up", lambda current_args: state)
    monkeypatch.setattr(orca_vast_host, "ensure_tunnel", lambda _: pytest.fail("ensure_tunnel should not run"))
    monkeypatch.setattr(orca_vast_host, "setup_existing_project", lambda *_: pytest.fail("setup_existing_project should not run"))

    assert orca_vast_host.handle_start_task(args) == 0
    output = capsys.readouterr().out
    assert "Vast host ready (bootstrap none): ssh-only" in output
    assert "ssh -i /tmp/id_ed25519 -p 2222" in output
    assert "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@203.0.113.10" in output
