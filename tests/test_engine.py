import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from moe_pilot.engine import Batch, Engine
from moe_pilot.smoke import tiny_model, verify_engine


def test_huggingface_numerical_parity_and_resume(tmp_path):
    report = verify_engine(tmp_path / "report.json")
    assert report["max_loss_error"] < 2e-5
    assert report["valid_tokens"] == 6
    assert json.loads((tmp_path / "report.json").read_text())["checkpoint_resume"] == "passed"


@pytest.mark.parametrize("normalize,clip", [(True, None), (False, 0.01), (True, 0.01)])
def test_router_normalization_and_qkv_clipping_match_reference(tmp_path, normalize, clip):
    model = tiny_model(tmp_path, norm_topk_prob=normalize, clip_qkv=clip)
    engine = Engine(tmp_path)
    batch = engine.batch([[2, 3, 4, 5]], 4)
    with torch.inference_mode():
        logits = model(batch.tok[None], use_cache=False).logits[0]
        expected = F.cross_entropy(logits[:-1], batch.tok[1:], reduction="none").numpy()
    losses, valid, _ = engine.run(batch, {}, verbose=False)
    np.testing.assert_allclose(losses["base"][valid], expected, atol=2e-5, rtol=2e-5)


def test_routing_masks_and_document_boundaries(local_model):
    _, engine = local_model
    batch = engine.batch([[2, 3, 4], [6, 7]], 8)
    results, valid, log = engine.run(batch, {"masked": {0: [0]}, "empty": {}}, log_routing=True)
    np.testing.assert_array_equal(valid, [True, True, False, True, False])
    np.testing.assert_array_equal(results["base"][[2, 4]], 0)
    np.testing.assert_array_equal(results["empty"], results["base"])
    assert len(log) == engine.L
    with torch.inference_mode():
        hidden = engine.get("model.embed_tokens.weight")[batch.tok]
        weights = engine.load_layer(0)
        masked_log = []
        engine.moe(engine.attn(hidden, weights, batch), weights, mask=[0], log=masked_log)
    assert not (masked_log[0][0] == 0).any()
    np.testing.assert_array_equal(masked_log[0][2][:, 0], 0)


def test_masked_losses_match_independent_reference_router(local_model):
    reference, engine = local_model
    batch = engine.batch([[2, 3, 4, 5]], 4)

    def mask_router(module, inputs, output):
        if isinstance(output, tuple):
            logits = output[0].clone()
            logits[:, 0] = -torch.inf
            weights, indices = logits.softmax(-1).topk(engine.K, dim=-1)
            if reference.config.norm_topk_prob:
                weights = weights / weights.sum(-1, keepdim=True)
            return logits, weights, indices
        logits = output.clone()
        logits[:, 0] = -torch.inf
        return logits

    hook = reference.model.layers[0].mlp.gate.register_forward_hook(mask_router)
    try:
        with torch.inference_mode():
            logits = reference(batch.tok[None], use_cache=False).logits[0]
            expected = F.cross_entropy(logits[:-1], batch.tok[1:], reduction="none").numpy()
    finally:
        hook.remove()
    results, valid, _ = engine.run(batch, {"masked": {0: [0]}}, verbose=False)
    np.testing.assert_allclose(results["masked"][valid], expected, atol=2e-5, rtol=2e-5)


def test_checkpoint_rejects_changed_model_shard(local_model, tmp_path):
    _, engine = local_model
    batch = engine.batch([[2, 3, 4]], 4)
    directory = tmp_path / "checkpoint"
    engine.run_ckpt(batch, {}, directory, verbose=False)
    (engine.store.directory / "model.safetensors").touch()
    with pytest.raises(ValueError, match="inputs"):
        engine.run_ckpt(batch, {}, directory)


def test_independent_engine_instances_and_no_global_thread_changes(local_model, tmp_path):
    reference, engine = local_model
    previous = torch.get_num_threads()
    reference.config.rms_norm_eps = 0.001
    reference.save_pretrained(tmp_path / "other")
    other = Engine(tmp_path / "other")
    assert other.cfg.rms_norm_eps != engine.cfg.rms_norm_eps
    assert torch.get_num_threads() == previous


def test_original_sharded_expert_format_matches_packed(local_model, tmp_path):
    reference, packed = local_model
    original = {}
    for name, value in reference.state_dict().items():
        if name.endswith(".experts.gate_up_proj"):
            prefix = name.removesuffix("gate_up_proj")
            for expert, tensor in enumerate(value):
                gate, up = tensor.chunk(2, dim=0)
                original[f"{prefix}{expert}.gate_proj.weight"] = gate.contiguous()
                original[f"{prefix}{expert}.up_proj.weight"] = up.contiguous()
        elif name.endswith(".experts.down_proj"):
            prefix = name.removesuffix("down_proj")
            for expert, tensor in enumerate(value):
                original[f"{prefix}{expert}.down_proj.weight"] = tensor.contiguous()
        else:
            original[name] = value.contiguous()
    directory = tmp_path / "original"
    directory.mkdir()
    reference.config.save_pretrained(directory)
    save_file(original, directory / "model-00001-of-00001.safetensors")
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(original, "model-00001-of-00001.safetensors")})
    )
    engine = Engine(directory)
    batch = Batch([[2, 3, 4], [5, 6]], 4)
    expected = packed.run(batch, {"mask": {1: [2]}}, verbose=False)[0]
    actual = engine.run(batch, {"mask": {1: [2]}}, verbose=False)[0]
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], atol=1e-6)


@pytest.mark.parametrize(
    "ids,limit", [([], 8), ([[]], 8), ([[1]], 8), ([[1, 2]], 1), ([[-1, 2]], 4)]
)
def test_invalid_batches_fail_early(ids, limit):
    with pytest.raises(ValueError):
        Batch(ids, limit)


@pytest.mark.parametrize(
    "conditions",
    [
        {"base": {}},
        {"../escape": {}},
        {"mask": {2: [0]}},
        {"mask": {0: [4]}},
        {"mask": {0: [0, 1, 2]}},
    ],
)
def test_invalid_conditions_fail_early(local_model, conditions):
    _, engine = local_model
    with pytest.raises(ValueError):
        engine.run(engine.batch([[2, 3]], 4), conditions)


def test_checkpoint_rejects_changed_inputs_and_masks(local_model, tmp_path):
    _, engine = local_model
    directory = tmp_path / "checkpoints"
    batch = engine.batch([[2, 3, 4]], 4)
    engine.run_ckpt(batch, {"mask": {0: [0]}}, directory, verbose=False)
    with pytest.raises(ValueError, match="condition"):
        engine.run_ckpt(batch, {"mask": {0: [1]}}, directory)
    with pytest.raises(ValueError, match="inputs"):
        engine.run_ckpt(engine.batch([[2, 3, 5]], 4), {}, directory)


def test_checkpoint_skips_completed_work_and_adds_new_conditions(
    local_model, tmp_path, monkeypatch
):
    _, engine = local_model
    batch = engine.batch([[2, 3, 4]], 4)
    directory = tmp_path / "checkpoints"
    engine.run_ckpt(batch, {"first": {1: [0]}}, directory, verbose=False)
    before = (directory / "losses/base.npy").stat().st_mtime_ns
    engine.run_ckpt(batch, {"first": {1: [0]}, "second": {0: [1]}}, directory, verbose=False)
    assert (directory / "losses/second.npy").is_file()
    # A new earlier condition may need a new baseline pass to populate its cache.
    assert (directory / "losses/base.npy").stat().st_mtime_ns >= before

    def unexpected(*args, **kwargs):
        pytest.fail("Completed checkpoint should not load model tensors")

    monkeypatch.setattr(engine, "get", unexpected)
    engine.run_ckpt(batch, {"first": {1: [0]}, "second": {0: [1]}}, directory)


def test_legacy_checkpoint_is_rejected(local_model, tmp_path):
    _, engine = local_model
    directory = tmp_path / "checkpoints"
    (directory / "losses").mkdir(parents=True)
    np.save(directory / "losses/base.npy", [0.0])
    with pytest.raises(ValueError, match="Legacy"):
        engine.run_ckpt(engine.batch([[2, 3]], 4), {}, directory)
