import json

import numpy as np
import pytest
import torch

from moe_pilot.verification import verify_local_model


def test_full_depth_local_reference_and_routing_are_verified(local_model, tmp_path):
    _, engine = local_model
    threads = torch.get_num_threads()
    output = tmp_path / "verification.json"
    report = verify_local_model(
        engine.store.directory,
        [[2, 3, 4, 5], [6, 7, 8]],
        max_tokens=4,
        device="cpu",
        output=output,
    )
    assert report["layers"] == engine.L
    assert report["parameters"] > 0
    assert report["valid_targets"] == 5
    assert report["max_loss_error"] < 2e-5
    assert report["router_set_agreement_by_layer"] == [1.0] * engine.L
    assert torch.get_num_threads() == threads
    assert json.loads(output.read_text())["status"] == "passed"


def test_failed_reference_parity_is_not_published(local_model, tmp_path, monkeypatch):
    _, engine = local_model
    from moe_pilot.engine import Engine

    original = Engine.run

    def altered(self, *args, **kwargs):
        results, valid, routes = original(self, *args, **kwargs)
        results["base"] += 0.1
        return results, valid, routes

    monkeypatch.setattr(Engine, "run", altered)
    output = tmp_path / "invalid.json"
    with pytest.raises(AssertionError):
        verify_local_model(engine.store.directory, [[2, 3, 4]], device="cpu", output=output)
    assert not output.exists()


def test_reference_uses_explicit_boundaries_and_truncation(local_model):
    _, engine = local_model
    report = verify_local_model(
        engine.store.directory, [[2, 3, 4, 5], [6, 7]], max_tokens=3, device="cpu"
    )
    assert report["input_tokens"] == 5
    assert report["valid_targets"] == 3
    assert np.isfinite(report["engine_mean_loss"])
