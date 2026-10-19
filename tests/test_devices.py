import json
from dataclasses import replace

import pytest
import torch

import moe_pilot.devices as devices
from moe_pilot.cli import main


def mock_backends(monkeypatch, *, cuda=False, hip=None, mps=False, count=1, failures=()):
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Test GPU")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    attempts = []

    def probe(device):
        attempts.append(str(device))
        if str(device) in failures:
            raise RuntimeError("Unsupported required kernel")

    monkeypatch.setattr(devices, "probe_device", probe)
    return attempts


def test_auto_prefers_cuda_before_mps_and_cpu(monkeypatch):
    attempts = mock_backends(monkeypatch, cuda=True, mps=True)
    selected = devices.resolve_device()
    assert selected.backend == "cuda"
    assert attempts == ["cuda:0"]


def test_rocm_uses_cuda_tensor_api_and_is_selected_before_mps(monkeypatch):
    attempts = mock_backends(monkeypatch, cuda=True, hip="7.2", mps=True)
    selected = devices.resolve_device()
    assert selected.backend == "rocm"
    assert str(selected.device) == "cuda:0"
    assert attempts == ["cuda:0"]


def test_auto_tries_mps_when_cuda_is_unavailable(monkeypatch):
    attempts = mock_backends(monkeypatch, mps=True)
    selected = devices.resolve_device()
    assert selected.backend == "mps"
    assert attempts == ["mps:0"]


def test_failed_cuda_probe_still_tries_mps_before_cpu(monkeypatch):
    attempts = mock_backends(monkeypatch, cuda=True, mps=True, failures=("cuda:0",))
    selected = devices.resolve_device()
    assert selected.backend == "mps"
    assert attempts == ["cuda:0", "mps:0"]
    assert "Unsupported required kernel" in selected.rejected[0]


def test_auto_tries_other_gpu_indices_before_lower_priority_backends(monkeypatch):
    attempts = mock_backends(monkeypatch, cuda=True, count=2, mps=True, failures=("cuda:0",))
    assert str(devices.resolve_device().device) == "cuda:1"
    assert attempts == ["cuda:0", "cuda:1"]


def test_cpu_is_used_only_after_accelerator_checks_fail(monkeypatch):
    attempts = mock_backends(monkeypatch, cuda=True, mps=True, failures=("cuda:0", "mps:0"))
    selected = devices.resolve_device()
    assert selected.backend == "cpu"
    assert attempts == ["cuda:0", "mps:0", "cpu"]
    assert len(selected.rejected) == 2


def test_explicit_device_does_not_silently_fall_back(monkeypatch):
    attempts = mock_backends(monkeypatch, cuda=True, mps=True, failures=("cuda:0",))
    with pytest.raises(ValueError, match="No compliant device"):
        devices.resolve_device("cuda:0")
    assert attempts == ["cuda:0"]


def test_explicit_rocm_never_uses_a_cuda_build(monkeypatch):
    mock_backends(monkeypatch, cuda=True)
    with pytest.raises(ValueError):
        devices.resolve_device("rocm")


@pytest.mark.parametrize("name", ["gpu", "cuda:-1", "mps:1", "rocm:wrong", ""])
def test_invalid_device_names_are_rejected(name):
    with pytest.raises(ValueError):
        devices.validate_device_name(name)


def test_cpu_capability_probe_checks_real_operations():
    devices.probe_device(torch.device("cpu"))


def test_devices_command_reports_why_fallback_was_needed(monkeypatch, capsys):
    mock_backends(monkeypatch)
    assert main(["devices"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["backend"] == "cpu"
    assert "mps" in report["rejected"][1]


def test_device_policy_is_validated_by_configuration(experiment):
    with pytest.raises(ValueError, match="device must"):
        replace(experiment, execution=replace(experiment.execution, device="tpu"))
