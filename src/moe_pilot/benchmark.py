"""Repeatable synchronized engine timings, independent of scientific results."""

import gc
import hashlib
import os
import platform
import statistics
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from transformers import OlmoeConfig

from moe_pilot.artifacts import write_json
from moe_pilot.devices import resolve_device, synchronize
from moe_pilot.engine import Engine
from moe_pilot.perturbations import Intervention
from moe_pilot.probes import OrthogonalityProbe
from moe_pilot.resources import ResourcePolicy, bounded_inference
from moe_pilot.smoke import synthetic_model, tiny_model

WORKLOADS = ("baseline", "routing_geometry", "ablations")


def _run(engine, batch, workload):
    conditions = (
        {"early": Intervention({0: [0]}), "late": Intervention({engine.L - 1: [1]})}
        if workload == "ablations"
        else {}
    )
    probe = (
        OrthogonalityProbe(np.arange(min(100, batch.N)), seed=17)
        if workload == "routing_geometry"
        else None
    )
    return engine.run(
        batch, conditions, log_routing=probe is not None, on_layer=probe, verbose=False
    )[:2]


def _check(actual, expected):
    losses, valid = actual
    reference, reference_valid = expected
    np.testing.assert_array_equal(valid, reference_valid)
    errors = []
    for name in reference:
        np.testing.assert_allclose(losses[name], reference[name], atol=2e-4, rtol=2e-4)
        errors.append(float(np.abs(losses[name] - reference[name]).max()))
    return max(errors)


def _measure(
    model_dir,
    *,
    devices,
    document_counts,
    max_tokens,
    warmup,
    repeats,
    threads,
    profile,
    weight_cache,
    workloads,
    resources,
):
    selections = {}
    for requested in devices:
        selection = resolve_device(requested)
        key = f"{selection.backend}:{selection.device}"
        selections[key] = selection
    rows, references = [], {}
    model = None
    # Only one model cache exists at a time. CPU runs require an explicit request
    # (or actual automatic fallback on a machine without a compliant accelerator).
    for selection in selections.values():
        engine = Engine(model_dir, device=selection, weight_cache=weight_cache, resources=resources)
        if max_tokens > engine.cfg.max_position_embeddings:
            raise ValueError("max_tokens exceeds the model context window")
        if "ablations" in workloads and engine.E <= engine.K:
            raise ValueError("Benchmark ablations require at least one maskable expert")
        if model is None:
            model = {
                "synthetic_profile": profile,
                "config": engine.cfg.to_dict(),
                "local_fingerprint": engine.store.fingerprint(),
                "weight_bytes": sum(
                    (Path(model_dir) / name).stat().st_size
                    for name in set(engine.store.weight_map.values())
                ),
            }
        for count in document_counts:
            rng = np.random.default_rng(17)
            ids = rng.integers(2, engine.cfg.vocab_size, size=(count, max_tokens)).tolist()
            batch = engine.batch(ids, max_tokens)
            for workload in workloads:
                case = (count, workload)
                elapsed, error = [], 0.0
                for _ in range(warmup):
                    actual = _run(engine, batch, workload)
                    if case not in references:
                        references[case] = actual
                    error = max(error, _check(actual, references[case]))
                for _ in range(repeats):
                    synchronize(engine.device)
                    start = perf_counter()
                    actual = _run(engine, batch, workload)
                    synchronize(engine.device)
                    elapsed.append(perf_counter() - start)
                    error = max(error, _check(actual, references[case]))
                median = statistics.median(elapsed)
                rows.append(
                    {
                        "workload": workload,
                        "documents": count,
                        "input_tokens": batch.N,
                        "valid_targets": int(references[case][1].sum()),
                        "backend": engine.selection.backend,
                        "device": str(engine.device),
                        "seconds": elapsed,
                        "median_seconds": median,
                        "min_seconds": min(elapsed),
                        "max_seconds": max(elapsed),
                        "input_tokens_per_second": batch.N / median,
                        "max_loss_error_vs_reference": error,
                        "resident_weight_bytes": engine.resident_weight_bytes,
                    }
                )
        device = engine.device
        del engine
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    for row in rows:
        cpu = next(
            (
                candidate
                for candidate in rows
                if candidate["backend"] == "cpu"
                and candidate["workload"] == row["workload"]
                and candidate["documents"] == row["documents"]
            ),
            None,
        )
        row["speedup_vs_cpu"] = cpu["median_seconds"] / row["median_seconds"] if cpu else None
    implementation = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    return {
        "schema_version": 2,
        "kind": "synthetic-engine-benchmark" if profile else "local-model-engine-benchmark",
        "recorded_at": datetime.now(UTC).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "threads": threads,
            "interop_threads": torch.get_num_interop_threads(),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "mps_cpu_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
        },
        "devices": [selection.record() for selection in selections.values()],
        "model": model,
        "method": {
            "warmup_runs_per_case_and_device": warmup,
            "measured_runs_per_case_and_device": repeats,
            "max_tokens": max_tokens,
            "weight_cache": weight_cache,
            "workloads": list(workloads),
            "document_counts": list(document_counts),
            "input_seed": 17,
            "timing": "synchronized wall time; requested backends run sequentially",
            "correctness_reference": next(iter(selections.values())).record(),
            "correctness": "repeat consistency and requested-backend agreement; not independent model parity",
            "resources": {
                "max_memory_gib": resources.max_memory_gib,
                "reserve_memory_gib": resources.reserve_memory_gib,
            },
            "included": "engine calls, layer reads/transfers, loss and host array transfers",
            "excluded": "imports, model creation, tokenization, device probes, JSON writes",
            "cache": "warm process and filesystem cache; no persisted result/checkpoint reuse",
            "routing_geometry": "routing plus all-layer geometry on up to 100 sampled tokens",
            "ablations": "baseline plus one early and one late single-expert mask",
            "throughput": "input tokens per engine call, not autoregressive generation tokens",
            "limits": "random token IDs; no corpus/report timing or scientific claims",
        },
        "implementation_sha256": implementation,
        "results": rows,
    }


@bounded_inference
def benchmark_engine(
    model_dir=None,
    *,
    profile="medium",
    devices=("auto",),
    document_counts=(1,),
    max_tokens=64,
    warmup=1,
    repeats=3,
    threads=2,
    output=None,
    weight_cache="stream",
    workloads=("baseline",),
    resources=None,
):
    """Measure requested devices only; never create a separate CPU model reference.

    With no model directory, construct a deterministic synthetic checkpoint.
    Library callers' process thread count is restored, including on failure.
    """
    if profile not in ("tiny", "medium"):
        raise ValueError("profile must be tiny or medium")
    if weight_cache not in ("stream", "resident"):
        raise ValueError("weight_cache must be stream or resident")
    if not workloads or len(workloads) != len(set(workloads)) or set(workloads) - set(WORKLOADS):
        raise ValueError("workloads must be unique supported benchmark workloads")
    for name, value, minimum in (
        ("max_tokens", max_tokens, 2),
        ("warmup", warmup, 1),
        ("repeats", repeats, 1),
        ("threads", threads, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not document_counts or any(type(n) is not int or n < 1 for n in document_counts):
        raise ValueError("document_counts must contain positive integers")
    if not devices:
        raise ValueError("devices must contain at least one backend")
    previous_threads = torch.get_num_threads()
    resources = resources or ResourcePolicy()
    resources.check()
    try:
        torch.set_num_threads(threads)
        with tempfile.TemporaryDirectory(prefix="moe-pilot-benchmark-") as temporary:
            if model_dir is None:
                model_dir = Path(temporary) / "model"
                if profile == "tiny":
                    tiny_model(model_dir)
                else:
                    config = OlmoeConfig(
                        vocab_size=1024,
                        hidden_size=512,
                        intermediate_size=768,
                        num_hidden_layers=2,
                        num_attention_heads=8,
                        num_key_value_heads=8,
                        num_experts=8,
                        num_experts_per_tok=2,
                        max_position_embeddings=max_tokens,
                        pad_token_id=0,
                        eos_token_id=1,
                        tie_word_embeddings=False,
                    )
                    synthetic_model(model_dir, config)
            else:
                profile = None
            report = _measure(
                Path(model_dir),
                devices=devices,
                document_counts=document_counts,
                max_tokens=max_tokens,
                warmup=warmup,
                repeats=repeats,
                threads=threads,
                profile=profile,
                weight_cache=weight_cache,
                workloads=workloads,
                resources=resources,
            )
        if output is not None:
            write_json(output, report)
        return report
    finally:
        torch.set_num_threads(previous_threads)
