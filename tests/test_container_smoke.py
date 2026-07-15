from __future__ import annotations

import json
import sys
import types

import pytest

from scripts.remote import container_smoke


_REQUIRED_RESULT_FIELDS = {
    "ok",
    "device",
    "cuda_available",
    "torch_version",
    "checks",
    "cwd",
}


def _stub_imports(monkeypatch: pytest.MonkeyPatch, failed: str | None = None) -> None:
    torch = types.ModuleType("torch")
    torch.__version__ = "test-torch"
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    classifier = types.ModuleType("src.classifier.model")
    classifier.LandscapeClassifier = object
    regression = types.ModuleType("src.scenic_scorer.regression")
    regression.ScenicRegressionModel = object
    modules = {
        "torch": torch,
        "torchvision": types.ModuleType("torchvision"),
        "timm": types.ModuleType("timm"),
        "boto3": types.ModuleType("boto3"),
        "src.classifier.model": classifier,
        "src.scenic_scorer.regression": regression,
    }

    def fake_import(module_path: str) -> types.ModuleType:
        if module_path == failed:
            raise ImportError(f"forced failure for {module_path}")
        return modules[module_path]

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(container_smoke.importlib, "import_module", fake_import)


def _run_cpu_smoke(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    failed: str | None = None,
) -> dict:
    _stub_imports(monkeypatch, failed=failed)
    monkeypatch.setattr(
        sys,
        "argv",
        ["container_smoke.py", "--device", "cpu", "--check-imports"],
    )
    container_smoke.main()
    output = capsys.readouterr().out.strip().splitlines()
    assert len(output) == 1
    return json.loads(output[0])


def test_check_import_failure_emits_false_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_imports(monkeypatch, failed="timm")
    monkeypatch.setattr(
        sys,
        "argv",
        ["container_smoke.py", "--device", "cpu", "--check-imports"],
    )

    with pytest.raises(SystemExit) as exc_info:
        container_smoke.main()

    output = capsys.readouterr().out.strip().splitlines()
    assert len(output) == 1
    result = json.loads(output[0])
    assert exc_info.value.code != 0
    assert set(result) == _REQUIRED_RESULT_FIELDS
    assert result["ok"] is False
    assert result["device"] == "cpu"
    assert result["cuda_available"] is False
    assert result["torch_version"] == "test-torch"
    assert result["checks"]["timm"] is False
    assert all(value for name, value in result["checks"].items() if name != "timm")


def test_cpu_import_check_success_preserves_json_contract(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = _run_cpu_smoke(monkeypatch, capsys)

    assert set(result) == _REQUIRED_RESULT_FIELDS
    assert result["ok"] is True
    assert result["device"] == "cpu"
    assert result["cuda_available"] is False
    assert result["torch_version"] == "test-torch"
    assert all(result["checks"].values())
