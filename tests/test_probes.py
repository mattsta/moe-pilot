import numpy as np
import torch

from moe_pilot.probes import OrthogonalityProbe


def test_geometry_observer_preserves_inference_and_global_rng(local_model):
    _, engine = local_model
    batch = engine.batch([[2, 3, 4, 5], [6, 7, 8]], 4)
    baseline, valid, _ = engine.run(batch, {}, verbose=False)
    before = torch.random.get_rng_state().clone()
    observer = OrthogonalityProbe([0, 1, 4], 17)
    observed, observed_valid, _ = engine.run(batch, {}, verbose=False, on_layer=observer)
    np.testing.assert_array_equal(observed_valid, valid)
    np.testing.assert_array_equal(observed["base"], baseline["base"])
    assert torch.equal(torch.random.get_rng_state(), before)
    assert len(observer.inputs) == 2
    assert len(observer.extra) == 2
    assert observer.matrices["w_0_gate_real"].shape == (4, 4)


def test_research_figures_do_not_register_pyplot_state():
    import matplotlib.pyplot as plt

    from moe_pilot.plotting import subplots

    previous = plt.get_fignums()
    figure, axes = subplots(figsize=(3, 2))
    axes.plot([0, 1], [0, 1])
    figure.canvas.draw()
    assert plt.get_fignums() == previous
