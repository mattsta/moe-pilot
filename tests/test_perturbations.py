import json
from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from moe_pilot.engine import Engine
from moe_pilot.perturbations import Intervention
from moe_pilot.smoke import tiny_model


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("mode", ["reroute", "remove", "rescale", "substitute"])
@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0])
def test_intervention_matches_independent_transformers_expert_execution(
    tmp_path, normalize, mode, strength
):
    reference = tiny_model(tmp_path / "model", norm_topk_prob=normalize)
    engine = Engine(tmp_path / "model", device="cpu")
    batch = engine.batch([[2, 3, 4, 5, 6, 7]], 6)
    spec = Intervention({0: [0]}, mode, strength, {0: {0: 3}} if mode == "substitute" else {})

    def alter_gate(module, inputs, result):
        logits, scores, indices = result
        if mode in ("remove", "rescale"):
            adjusted = scores * torch.where(indices == 0, 1 - strength, 1)
            if mode == "rescale":
                retained = adjusted.sum(-1, keepdim=True)
                adjusted *= scores.sum(-1, keepdim=True) / retained.clamp_min(1e-30)
            return logits, adjusted, indices
        if mode == "substitute":
            new_indices = torch.where(indices == 0, 3, indices)
            new_scores = scores
        else:
            modified = logits.clone()
            modified[:, 0] = -torch.inf
            new_scores, new_indices = modified.softmax(-1).topk(engine.K, dim=-1)
            if normalize:
                new_scores /= new_scores.sum(-1, keepdim=True)
        # Transformers executes both sets of original and altered contributions.
        # This independently checks strength interpolation in expert-output space.
        return (
            logits,
            torch.cat((scores * (1 - strength), new_scores * strength), dim=-1),
            torch.cat((indices, new_indices), dim=-1),
        )

    hook = reference.model.layers[0].mlp.gate.register_forward_hook(alter_gate)
    try:
        with torch.inference_mode():
            logits = reference(batch.tok[None], use_cache=False).logits[0]
            expected = F.cross_entropy(logits[:-1], batch.tok[1:], reduction="none").numpy()
    finally:
        hook.remove()
    actual, valid, _ = engine.run(batch, {"condition": spec}, verbose=False)
    np.testing.assert_allclose(actual["condition"][valid], expected, atol=2e-5, rtol=2e-5)
    if strength == 0:
        np.testing.assert_array_equal(actual["condition"], actual["base"])


def test_all_removed_rescaling_has_explicit_finite_zero_output(local_model):
    _, engine = local_model
    batch = engine.batch([[2, 3, 4, 5]], 4)
    records = []
    result = engine.run(
        batch,
        {"all": Intervention({0: [0, 1, 2, 3]}, "rescale")},
        verbose=False,
        on_intervention=lambda name, layer, record: records.append(record),
    )[0]
    assert np.isfinite(result["all"]).all()
    assert records[0]["zero_post_mass_tokens"] == batch.N
    assert records[0]["post_selected_mass_mean"] == 0
    assert records[0]["output_delta_norm_mean"] > 0


def test_checkpoint_records_diagnostics_and_rejects_changed_semantics(local_model, tmp_path):
    _, engine = local_model
    batch = engine.batch([[2, 3, 4, 5]], 4)
    directory = tmp_path / "checkpoint"
    spec = Intervention({0: [0]}, "remove")
    engine.run_ckpt(batch, {"target": spec}, directory, record_diagnostics=True, verbose=False)
    records = json.loads((directory / "diagnostics/target.json").read_text())
    assert records[0]["mode"] == "remove"
    assert records[0]["fraction_changed_expert_sets"] == 0
    assert records[0]["post_selected_mass_mean"] <= records[0]["original_selected_mass_mean"]
    before = (directory / "losses/target.npy").stat().st_mtime_ns
    engine.run_ckpt(batch, {"target": spec}, directory, record_diagnostics=True, verbose=False)
    assert (directory / "losses/target.npy").stat().st_mtime_ns == before
    for changed in (replace(spec, mode="rescale"), replace(spec, strength=0.5)):
        with pytest.raises(ValueError, match="condition"):
            engine.run_ckpt(batch, {"target": changed}, directory, verbose=False)


def test_missing_checkpoint_diagnostics_are_recomputed(local_model, tmp_path):
    _, engine = local_model
    batch = engine.batch([[2, 3, 4]], 4)
    directory = tmp_path / "checkpoint"
    conditions = {"target": Intervention({0: [0]}, "substitute", donors={0: {0: 3}})}
    engine.run_ckpt(batch, conditions, directory, verbose=False)
    assert not (directory / "diagnostics/target.json").exists()
    engine.run_ckpt(batch, conditions, directory, record_diagnostics=True, verbose=False)
    assert (
        json.loads((directory / "diagnostics/target.json").read_text())[0]["mode"] == "substitute"
    )
    with pytest.raises(ValueError, match="condition"):
        engine.run_ckpt(
            batch,
            {"target": Intervention({0: [0]}, "substitute", donors={0: {0: 2}})},
            directory,
            verbose=False,
        )


def test_interventions_are_deeply_immutable_and_replaceable():
    mask, donors = {0: [0]}, {0: {0: 3}}
    spec = Intervention(mask, "substitute", donors=donors)
    mask[0].append(1)
    donors[0][0] = 2
    assert spec.mask[0] == (0,)
    assert spec.donors[0][0] == 3
    assert replace(spec, strength=0.5).strength == 0.5
    with pytest.raises(TypeError):
        spec.donors[0][0] = 1


@pytest.mark.parametrize(
    "options",
    [
        {"mode": "unknown"},
        {"strength": float("nan")},
        {"strength": True},
        {"strength": -0.1},
        {"mask": {0: [0, 0]}},
        {"mask": {True: [0]}},
        {"mode": "substitute"},
        {"mode": "substitute", "donors": {0: {0: 0}}},
        {"donors": {0: {0: 3}}},
    ],
)
def test_invalid_interventions_fail_before_inference(options):
    with pytest.raises(ValueError):
        Intervention(**{"mask": {0: [0]}, **options})
