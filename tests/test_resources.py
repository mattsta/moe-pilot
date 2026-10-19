from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from filelock import FileLock

from moe_pilot.engine import Engine
from moe_pilot.resources import GIB, ResourcePolicy


@pytest.fixture
def memory(monkeypatch):
    from moe_pilot import resources

    monkeypatch.setattr(resources, "physical_memory_bytes", lambda: 128 * GIB)
    monkeypatch.setattr(resources, "available_memory_bytes", lambda: 64 * GIB)
    monkeypatch.setattr(resources, "peak_process_bytes", lambda: GIB)
    return resources


def test_oversized_job_rejected_before_any_tensor_load(local_model, memory, monkeypatch):
    _, source = local_model
    engine = Engine(source.store.directory, device="cpu", weight_cache="resident")
    monkeypatch.setattr(engine.store, "tensor_bytes", lambda name: 2 * GIB)

    def forbidden(*args):
        pytest.fail("Weights must not be loaded before resource admission")

    monkeypatch.setattr(engine.store, "get", forbidden)
    with pytest.raises(ValueError, match="budget"):
        engine.run(engine.batch([[2, 3, 4]], 4), {})


def test_pressure_stops_execution_instead_of_switching_to_cpu(memory):
    policy = ResourcePolicy()
    memory.available_memory_bytes = lambda: 17 * GIB
    with pytest.raises(ValueError, match="workstation reserve"):
        policy.check(2 * GIB)


def test_process_over_budget_is_stopped(memory):
    policy = ResourcePolicy()
    memory.peak_process_bytes = lambda: 9 * GIB
    with pytest.raises(ValueError, match="Process memory"):
        policy.check()


def test_small_host_preserves_quarter_of_memory(memory):
    memory.physical_memory_bytes = lambda: 8 * GIB
    assert ResourcePolicy().reserve_bytes == 2 * GIB


def test_mps_allocator_limit_is_per_process(memory, monkeypatch):
    limits = []
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 64 * GIB)
    monkeypatch.setattr(torch.mps, "set_per_process_memory_fraction", limits.append)
    ResourcePolicy().limit_device(torch.device("mps"))
    assert limits == [0.125]


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_resource_budgets_rejected(value):
    with pytest.raises(ValueError):
        ResourcePolicy(max_memory_gib=value)


def test_configuration_propagates_resource_budgets(experiment, memory):
    config = replace(experiment, execution=replace(experiment.execution, max_memory_gib=3))
    assert config.execution.resource_policy().max_memory_gib == 3


def test_concurrent_job_rejected_and_nested_calls_allowed(tmp_path, monkeypatch):
    from moe_pilot import resources

    filename = tmp_path / "compute.lock"
    monkeypatch.setattr(resources, "_COMPUTE_LOCK", FileLock(filename))

    @resources.bounded_inference
    def inner():
        return 42

    @resources.bounded_inference
    def outer():
        return inner()

    with FileLock(filename):
        with pytest.raises(ValueError, match="concurrent inference"):
            outer()
    assert outer() == 42


def test_reference_rejected_before_transformers_load(local_model, memory, monkeypatch):
    from moe_pilot.model import ModelStore
    from moe_pilot.verification import AutoModelForCausalLM, verify_local_model

    _, engine = local_model
    monkeypatch.setattr(ModelStore, "tensor_bytes", lambda self, name: GIB)

    def forbidden(*args, **kwargs):
        pytest.fail("An oversized reference must not be constructed")

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", forbidden)
    with pytest.raises(ValueError, match="budget"):
        verify_local_model(engine.store.directory, [[2, 3, 4]], device="cpu")


def test_macos_headroom_counts_file_cache_once_but_never_anonymous_pages(monkeypatch):
    from moe_pilot import resources

    monkeypatch.setattr(resources.platform, "system", lambda: "Darwin")
    stats = "Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages free: 100.\nPages speculative: 20.\nPages inactive: 1000.\nFile-backed pages: 200.\nAnonymous pages: 50000.\n"
    monkeypatch.setattr(
        resources.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(
            stdout="1" if command[0].endswith("sysctl") else stats
        ),
    )
    assert resources.available_memory_bytes() == 300 * 4096


def test_macos_warning_pressure_stops_before_reading_cache_counters(monkeypatch):
    from moe_pilot import resources

    monkeypatch.setattr(resources.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        resources.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="2")
    )
    with pytest.raises(ValueError, match="macOS reports memory pressure"):
        resources.available_memory_bytes()
