"""Thin CLI over the same explicit experiment API used by library callers."""

import argparse
import json
import re
import sys
from pathlib import Path

from moe_pilot.config import load_experiment
from moe_pilot.experiment import STEPS, ExperimentRunner


def parser():
    result = argparse.ArgumentParser(description="Configured MoE research experiments")
    result.add_argument(
        "--config", type=Path, help="Experiment TOML; not needed for smoke, devices, or benchmark"
    )
    commands = result.add_subparsers(dest="command", required=True)
    for name in STEPS:
        commands.add_parser(name, help=f"Run the {name} experiment step")
    commands.add_parser("run", help="Run inference, conditions, ablations, and reports in order")
    commands.add_parser("download-model", help="Download the exact configured model revision")
    commands.add_parser("inventory", help="Read configured input and output availability")
    verification = commands.add_parser("verify-model", help="Full-depth local CPU reference parity")
    verification.add_argument("--documents", type=int, default=5)
    verification.add_argument("--max-tokens", type=int)
    verification.add_argument("--output", type=Path, required=True)
    smoke = commands.add_parser("smoke", help="Offline tiny-model numerical verification")
    smoke.add_argument("--output", type=Path, help="Optional JSON report destination")
    smoke.add_argument(
        "--device", default="auto", help="auto, cuda[:index], rocm[:index], mps, or cpu"
    )
    devices = commands.add_parser(
        "devices", help="Probe available backends and report the selected device"
    )
    devices.add_argument("--device", default="auto", help="Requested device (default: auto)")
    benchmark = commands.add_parser("benchmark", help="Compare synchronized engine performance")
    benchmark.add_argument("--model", type=Path, help="Local model; omit for synthetic weights")
    benchmark.add_argument("--profile", choices=("tiny", "medium"), default="medium")
    benchmark.add_argument("--devices", nargs="+", default=["cpu", "auto"])
    benchmark.add_argument("--documents", type=int, nargs="+", default=[1, 16])
    benchmark.add_argument("--max-tokens", type=int, default=64)
    benchmark.add_argument("--warmup", type=int, default=2)
    benchmark.add_argument("--repeats", type=int, default=5)
    benchmark.add_argument("--threads", type=int, default=8)
    benchmark.add_argument("--output", type=Path, required=True, help="JSON measurement record")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "benchmark":
            from moe_pilot.benchmark import benchmark_engine

            report = benchmark_engine(
                arguments.model,
                profile=arguments.profile,
                devices=arguments.devices,
                document_counts=arguments.documents,
                max_tokens=arguments.max_tokens,
                warmup=arguments.warmup,
                repeats=arguments.repeats,
                threads=arguments.threads,
                output=arguments.output,
            )
            print("workload documents backend median_seconds speedup_vs_cpu")
            for row in report["results"]:
                ratio = row["speedup_vs_cpu"]
                speedup = f"{ratio:.2f}x" if ratio is not None else "n/a"
                print(
                    f"{row['workload']} {row['documents']} {row['backend']} "
                    f"{row['median_seconds']:.6f} {speedup}"
                )
            print(f"Measurement record: {arguments.output}")
            return 0
        if arguments.command == "devices":
            from moe_pilot.devices import resolve_device

            print(json.dumps(resolve_device(arguments.device).record(), indent=2))
            return 0
        if arguments.command == "smoke":
            from moe_pilot.smoke import verify_engine

            report = verify_engine(arguments.output, device=arguments.device)
            print(f"Offline smoke passed; max loss error {report['max_loss_error']:.3g}")
            return 0
        if arguments.config is None:
            parser().error("--config is required for experiment commands")
        config = load_experiment(arguments.config)
        if arguments.command == "inventory":
            from moe_pilot.inventory import inspect_experiment

            print(inspect_experiment(config))
            return 0
        if arguments.command == "download-model":
            if not re.fullmatch(r"[0-9a-fA-F]{40}", config.model.revision):
                raise ValueError("model.revision must be a full 40-character commit SHA")
            from huggingface_hub import snapshot_download

            from moe_pilot.artifacts import write_json

            snapshot_download(
                config.model.repository,
                revision=config.model.revision,
                local_dir=config.paths.model,
                allow_patterns=["*.json", "*.safetensors", "*.txt"],
            )
            write_json(
                config.paths.model / "model_revision.json",
                {"repository": config.model.repository, "revision": config.model.revision},
            )
            return 0
        import torch

        torch.set_num_threads(config.execution.threads)
        if arguments.command == "verify-model":
            from moe_pilot.artifacts import load_corpus
            from moe_pilot.engine import Engine
            from moe_pilot.verification import verify_local_model

            if arguments.documents < 1:
                raise ValueError("documents must be positive")
            corpus = load_corpus(config.paths.data / "corpus.json")
            # First represent each available domain, then add remaining rows.
            chosen, seen = [], set()
            for index, record in enumerate(corpus):
                if record["domain"] not in seen:
                    chosen.append(index)
                    seen.add(record["domain"])
            chosen.extend(index for index in range(len(corpus)) if index not in chosen)
            engine = Engine(config.paths.model, device=config.execution.device)
            report = verify_local_model(
                config.paths.model,
                [
                    engine.tok(corpus[index]["text"]).input_ids
                    for index in chosen[: arguments.documents]
                ],
                max_tokens=(
                    config.execution.max_tokens
                    if arguments.max_tokens is None
                    else arguments.max_tokens
                ),
                device=engine.selection,
                output=arguments.output,
            )
            print(f"Full-depth parity passed; max loss error {report['max_loss_error']:.3g}")
            return 0
        runner = ExperimentRunner(config)
        print(f"Compute device: {runner.selection.backend} ({runner.selection.device})")
        if arguments.command == "run":
            runner.run_pipeline()
        else:
            runner.run(arguments.command)
    except (OSError, ValueError) as error:
        print(f"moe-pilot: {error}", file=sys.stderr)
        return 1
    return 0
