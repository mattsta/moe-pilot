"""Explicit full-depth parity validation on the selected compute device."""

import gc
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from moe_pilot.artifacts import write_json
from moe_pilot.devices import synchronize, to_numpy
from moe_pilot.engine import Engine
from moe_pilot.resources import bounded_inference


@bounded_inference
@torch.inference_mode()
def verify_local_model(
    model_dir,
    token_ids,
    *,
    max_tokens=64,
    device="auto",
    weight_cache="stream",
    resources=None,
    output=None,
):
    """Check every layer and valid next-token target on explicit local token IDs.

    The selected-device reference is released before engine inference. No weights
    or tokenizers are downloaded, and process-wide thread settings are unchanged.
    Router-set agreement is reported separately from loss tolerance.
    """
    engine = Engine(model_dir, device=device, weight_cache=weight_cache, resources=resources)
    selection = engine.selection
    batch = engine.batch(token_ids, max_tokens)
    identity = engine.store.fingerprint()
    # Reject a full reference that exceeds the shared-workstation budget before
    # Transformers allocates any parameters. Conversion/staging can double weights.
    reference_bytes = sum(engine.store.tensor_bytes(name) for name in engine.store.weight_map)
    engine.resources.check(reference_bytes * 2 + batch.N * engine.cfg.vocab_size * 12)
    start = perf_counter()
    print(f"Loading full-depth reference on {selection.backend} from local weights", flush=True)
    reference = (
        AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            dtype=torch.float32,
            trust_remote_code=False,
        )
        .to(selection.device)
        .eval()
    )
    parameters = sum(parameter.numel() for parameter in reference.parameters())
    expected, routes = [], []
    for index, document in enumerate(batch.ids):
        engine.resources.check()
        tokens = torch.tensor([document], device=selection.device)
        result = reference(tokens, use_cache=False, output_router_logits=True)
        losses = F.cross_entropy(result.logits[0, :-1], tokens[0, 1:], reduction="none")
        expected.extend(to_numpy(losses).tolist() + [0.0])
        routes.append(
            [
                to_numpy(logits.float().softmax(-1).topk(engine.K, dim=-1).indices)
                for logits in result.router_logits
            ]
        )
        print(f"{selection.backend} reference document {index + 1}/{len(batch.ids)}", flush=True)
        del result, tokens, losses
    reference_seconds = perf_counter() - start
    del reference
    gc.collect()
    if selection.device.type == "mps":
        torch.mps.empty_cache()
    elif selection.device.type == "cuda":
        torch.cuda.empty_cache()
    synchronize(selection.device)
    start = perf_counter()
    actual, valid, routing = engine.run(batch, {}, log_routing=True, verbose=False)
    synchronize(selection.device)
    engine_seconds = perf_counter() - start
    expected = np.array(expected)
    tolerance = 2e-5 if selection.backend == "cpu" else 2e-4
    np.testing.assert_allclose(actual["base"], expected, atol=tolerance, rtol=tolerance)
    agreement = []
    for layer in range(engine.L):
        reference_routes = np.concatenate([document[layer] for document in routes])
        matching = np.all(
            np.sort(reference_routes, axis=-1) == np.sort(routing[layer][0], axis=-1), axis=-1
        )
        agreement.append(float(matching.mean()))
    if engine.store.fingerprint() != identity:
        raise ValueError("Local model changed during full-depth verification")
    report = {
        "schema_version": 1,
        "kind": "full-depth-local-model-verification",
        "status": "passed",
        "model_fingerprint": identity,
        "model_config": engine.cfg.to_dict(),
        "parameters": parameters,
        "layers": engine.L,
        "experts": engine.E,
        "top_k": engine.K,
        "documents": len(batch.ids),
        "input_tokens": batch.N,
        "valid_targets": int(valid.sum()),
        "token_ids_sha256": hashlib.sha256(json.dumps(batch.ids).encode()).hexdigest(),
        "device": selection.record(),
        "versions": {name: version(name) for name in ("torch", "transformers", "safetensors")},
        "threads": torch.get_num_threads(),
        "weight_cache": weight_cache,
        "resident_weight_bytes": engine.resident_weight_bytes,
        "max_loss_error": float(np.abs(actual["base"] - expected).max()),
        "atol": tolerance,
        "rtol": tolerance,
        "reference_mean_loss": float(expected[valid].mean()),
        "engine_mean_loss": float(actual["base"][valid].mean()),
        "router_set_agreement_by_layer": agreement,
        "timing": {
            "reference_load_and_documents_seconds": reference_seconds,
            "streaming_engine_seconds": engine_seconds,
            "comparison": "different timing boundaries; not a performance speedup measurement",
        },
        "implementation_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    if output is not None:
        write_json(output, report)
    return report
