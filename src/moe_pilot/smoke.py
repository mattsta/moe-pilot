"""Reusable offline integration check. Synthetic results are not research evidence."""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import OlmoeConfig, OlmoeForCausalLM

from moe_pilot.devices import resolve_device
from moe_pilot.engine import Engine


def synthetic_model(directory, config):
    """Save deterministic CPU-generated weights without changing global RNG state."""
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(42)
        model = OlmoeForCausalLM(config).eval()
    model.save_pretrained(directory, safe_serialization=True)
    return model


def tiny_model(directory, *, layers=2, experts=4, norm_topk_prob=False, clip_qkv=None):
    """Create a deterministic local test checkpoint without network access."""
    config = OlmoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=experts,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        pad_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        norm_topk_prob=norm_topk_prob,
        clip_qkv=clip_qkv,
    )
    return synthetic_model(directory, config)


@torch.inference_mode()
def verify_engine(output=None, *, device="auto"):
    """Compare streaming losses with Hugging Face and verify checkpoint resume."""
    with tempfile.TemporaryDirectory(prefix="moe-pilot-smoke-") as temporary:
        root = Path(temporary)
        model = tiny_model(root / "model")
        selection = resolve_device(device)
        engine = Engine(root / "model", device=selection)
        batch = engine.batch([[2, 3, 4, 5, 6], [7, 8, 9]], max_tokens=8)
        conditions = {"masked": {0: [0]}, "unmasked": {}}
        results, valid, routing = engine.run(batch, conditions, log_routing=True, verbose=False)
        expected = []
        for document in batch.ids:
            tokens = torch.tensor([document])
            logits = model(tokens, use_cache=False).logits[0]
            losses = F.cross_entropy(logits[:-1], tokens[0, 1:], reduction="none")
            expected.extend(losses.tolist() + [0.0])
        error = float(np.max(np.abs(results["base"] - expected)))
        tolerance = 2e-5 if selection.backend == "cpu" else 2e-4
        np.testing.assert_allclose(results["base"], expected, atol=tolerance, rtol=tolerance)
        np.testing.assert_array_equal(results["unmasked"], results["base"])
        if np.any(routing[0][0] < 0):
            raise AssertionError("Invalid routing index")
        checkpoint = root / "checkpoint"
        engine.run_ckpt(batch, conditions, checkpoint, verbose=False)
        for name, losses in results.items():
            np.testing.assert_allclose(
                np.load(checkpoint / f"losses/{name}.npy"), losses, atol=1e-6
            )
        np.testing.assert_array_equal(np.load(checkpoint / "valid.npy"), valid)
        engine.run_ckpt(batch, conditions, checkpoint, verbose=False)
        report = {
            "kind": "synthetic-software-check",
            "max_loss_error": error,
            "valid_tokens": int(valid.sum()),
            "conditions": list(results),
            "torch": torch.__version__,
            "device": selection.record(),
            "checkpoint_resume": "passed",
        }
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
    return report
