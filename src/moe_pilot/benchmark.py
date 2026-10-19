"""Repeatable synchronized engine timings, independent of scientific results."""

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


def _measure(model_dir, *, devices, document_counts, max_tokens, warmup, repeats, threads, profile):
    reference_engine = Engine(model_dir, device="cpu")
    if max_tokens > reference_engine.cfg.max_position_embeddings:
        raise ValueError("max_tokens exceeds the model context window")
    if reference_engine.E <= reference_engine.K:
        raise ValueError("Benchmark ablations require at least one maskable expert")
    engines = {}
    for requested in devices:
        selection = resolve_device(requested)
        key = f"{selection.backend}:{selection.device}"
        if key not in engines:
            engines[key] = Engine(model_dir, device=selection)
    rows = []
    for count in document_counts:
        rng = np.random.default_rng(17)
        ids = rng.integers(2, reference_engine.cfg.vocab_size, size=(count, max_tokens)).tolist()
        batch = reference_engine.batch(ids, max_tokens)
        for workload in WORKLOADS:
            expected = _run(reference_engine, batch, workload)
            samples = {key: [] for key in engines}
            errors = dict.fromkeys(engines, 0.0)
            for engine in engines.values():
                for _ in range(warmup):
                    _check(_run(engine, batch, workload), expected)
                synchronize(engine.device)
            # Alternate backend order to reduce systematic drift during a run.
            for iteration in range(repeats):
                order = list(engines.items())
                if iteration % 2:
                    order.reverse()
                for key, engine in order:
                    synchronize(engine.device)
                    start = perf_counter()
                    actual = _run(engine, batch, workload)
                    synchronize(engine.device)
                    samples[key].append(perf_counter() - start)
                    errors[key] = max(errors[key], _check(actual, expected))
            cpu_median = next(
                (
                    statistics.median(samples[key])
                    for key, engine in engines.items()
                    if engine.selection.backend == "cpu"
                ),
                None,
            )
            for key, engine in engines.items():
                elapsed = samples[key]
                median = statistics.median(elapsed)
                rows.append(
                    {
                        "workload": workload,
                        "documents": count,
                        "input_tokens": batch.N,
                        "valid_targets": int(expected[1].sum()),
                        "backend": engine.selection.backend,
                        "device": str(engine.device),
                        "seconds": elapsed,
                        "median_seconds": median,
                        "min_seconds": min(elapsed),
                        "max_seconds": max(elapsed),
                        "input_tokens_per_second": batch.N / median,
                        "speedup_vs_cpu": cpu_median / median if cpu_median else None,
                        "max_loss_error_vs_cpu": errors[key],
                    }
                )
    implementation = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    return {
        "schema_version": 1,
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
        "devices": [engine.selection.record() for engine in engines.values()],
        "model": {
            "synthetic_profile": profile,
            "config": reference_engine.cfg.to_dict(),
            "local_fingerprint": reference_engine.store.fingerprint(),
            "weight_bytes": sum(
                (Path(model_dir) / name).stat().st_size
                for name in set(reference_engine.store.weight_map.values())
            ),
        },
        "method": {
            "warmup_runs_per_case_and_device": warmup,
            "measured_runs_per_case_and_device": repeats,
            "max_tokens": max_tokens,
            "document_counts": list(document_counts),
            "input_seed": 17,
            "timing": "synchronized wall time; backend order alternates per repeat",
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


def benchmark_engine(
    model_dir=None,
    *,
    profile="medium",
    devices=("cpu", "auto"),
    document_counts=(1, 16),
    max_tokens=64,
    warmup=2,
    repeats=5,
    threads=8,
    output=None,
):
    """Compare the same model/inputs across devices; fail on numerical disagreement.

    With no model directory, construct a deterministic synthetic checkpoint.
    Library callers' process thread count is restored, including on failure.
    """
    if profile not in ("tiny", "medium"):
        raise ValueError("profile must be tiny or medium")
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
            )
        if output is not None:
            write_json(output, report)
        return report
    finally:
        torch.set_num_threads(previous_threads)
