"""Numerical and artifact tests on real accelerators, when the host exposes them."""

import numpy as np
import pytest
import torch

from moe_pilot.devices import resolve_device
from moe_pilot.engine import Engine
from moe_pilot.probes import OrthogonalityProbe
from moe_pilot.smoke import tiny_model, verify_engine


@pytest.mark.parametrize("backend", ["cuda", "rocm", "mps"])
def test_accelerator_inference_geometry_and_checkpoint_roundtrip(backend, tmp_path):
    if backend == "mps":
        available = torch.backends.mps.is_available()
    else:
        available = torch.cuda.is_available() and bool(torch.version.hip) == (backend == "rocm")
    if not available:
        pytest.skip(f"Host has no {backend} backend")
    # An advertised backend that fails capability validation is a failure,
    # rather than a skip that could hide unsupported operations.
    selected = resolve_device(backend)
    report = verify_engine(tmp_path / "smoke.json", device=backend)
    assert report["device"]["backend"] == backend
    model_dir = tmp_path / "model"
    tiny_model(model_dir)
    engine = Engine(model_dir, device=selected)
    cpu = Engine(model_dir, device="cpu")
    batch = engine.batch([[2, 3, 4, 5], [6, 7, 8]], 6)
    conditions = {"masked": {0: [0]}, "late": {1: [1]}}
    observer = OrthogonalityProbe([0, 1, 4], 17)
    actual, valid, routing = engine.run(
        batch, conditions, log_routing=True, on_layer=observer, verbose=False
    )
    expected, expected_valid, _ = cpu.run(batch, conditions, verbose=False)
    np.testing.assert_array_equal(valid, expected_valid)
    for name in actual:
        np.testing.assert_allclose(actual[name], expected[name], atol=2e-4, rtol=2e-4)
    assert len(observer.extra) == 2
    assert all(np.isfinite(value).all() for value in observer.matrices.values())
    assert routing[0][0].dtype == np.int32
    weights = engine.load_layer(0)
    assert weights["G"][0].device == selected.device
    checkpoint = tmp_path / "checkpoint"
    engine.run_ckpt(batch, conditions, checkpoint, verbose=False)
    for name in actual:
        np.testing.assert_allclose(
            np.load(checkpoint / f"losses/{name}.npy"), actual[name], atol=2e-5, rtol=2e-5
        )
    engine.run_ckpt(batch, conditions, checkpoint, verbose=False)
    with pytest.raises(ValueError, match="inputs changed"):
        cpu.run_ckpt(batch, conditions, checkpoint, verbose=False)
