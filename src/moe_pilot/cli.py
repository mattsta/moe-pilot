"""Thin CLI over the same explicit experiment API used by library callers."""

import argparse
import json
import sys
from pathlib import Path

from moe_pilot.config import load_experiment
from moe_pilot.experiment import STEPS, ExperimentRunner


def parser():
    result = argparse.ArgumentParser(description="Configured MoE research experiments")
    result.add_argument(
        "--config", type=Path, help="Experiment TOML for configured research commands"
    )
    commands = result.add_subparsers(dest="command", required=True)
    for name in STEPS:
        commands.add_parser(name, help=f"Run the {name} experiment step")
    commands.add_parser("run", help="Run inference, conditions, ablations, and reports in order")
    commands.add_parser("download-model", help="Download the exact configured model revision")
    catalog = commands.add_parser(
        "download-models", help="Acquire named immutable local model targets"
    )
    catalog.add_argument("--catalog", type=Path, required=True)
    catalog.add_argument("--names", nargs="+", required=True)
    commands.add_parser("download-study-data", help="Acquire pinned local study source files")
    audit = commands.add_parser(
        "inspect-study-data", help="Audit local source origins and eligible cells"
    )
    audit.add_argument("--output", type=Path, required=True)
    publication = commands.add_parser(
        "publish-reconstruction", help="Validate and export local research evidence"
    )
    publication.add_argument("--output", type=Path, required=True)
    commands.add_parser("inventory", help="Read configured input and output availability")
    report = commands.add_parser(
        "report", help="Build the offline interactive HTML research report"
    )
    report.add_argument("--reconstruction", type=Path, default=Path("research/reconstruction-0924"))
    report.add_argument("--study", type=Path, default=Path("runs/causal-study"))
    report.add_argument("--output", type=Path, default=Path("research/report"))
    preview = commands.add_parser(
        "serve-report", help="Serve the report locally without model imports"
    )
    preview.add_argument("--directory", type=Path, default=Path("research/report"))
    preview.add_argument("--port", type=int, default=8765)
    preview.add_argument("--study", type=Path, default=Path("runs/causal-study"))
    study = commands.add_parser("study-observe", help="Resume bounded local research observations")
    study.add_argument("--catalog", type=Path, required=True)
    study.add_argument("--names", nargs="+", required=True)
    study.add_argument(
        "--partition",
        choices=("discovery", "calibration", "evaluation", "format_pairs"),
        required=True,
    )
    study.add_argument("--max-batches", type=int)
    study.add_argument("--interventions", action="store_true")
    study.add_argument("--formatting", choices=("plain", "paired-chat"), default="plain")
    commands.add_parser("study-select", help="Freeze controls from discovery and calibration only")
    commands.add_parser("study-analyse", help="Analyse completed research observations")
    reanalysis = commands.add_parser(
        "analyse-reconstruction", help="Compute paired on/off interactions from existing research"
    )
    reanalysis.add_argument(
        "--reconstruction", type=Path, default=Path("research/reconstruction-0924")
    )
    reanalysis.add_argument("--output", type=Path, default=Path("runs/causal-study/tables"))
    verification = commands.add_parser(
        "verify-model", help="Explicit, memory-budgeted model parity"
    )
    verification.add_argument("--documents", type=int, default=5)
    verification.add_argument("--max-tokens", type=int)
    verification.add_argument(
        "--partition", choices=("corpus", "discovery", "calibration"), default="corpus"
    )
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
    benchmark.add_argument("--devices", nargs="+", default=["auto"])
    benchmark.add_argument("--documents", type=int, nargs="+", default=[1])
    benchmark.add_argument("--max-tokens", type=int, default=64)
    benchmark.add_argument("--warmup", type=int, default=1)
    benchmark.add_argument("--repeats", type=int, default=3)
    benchmark.add_argument("--threads", type=int, default=2)
    benchmark.add_argument("--max-memory-gib", type=int, default=8)
    benchmark.add_argument("--reserve-memory-gib", type=int, default=16)
    benchmark.add_argument("--weight-cache", choices=("stream", "resident"), default="stream")
    benchmark.add_argument(
        "--workloads",
        nargs="+",
        choices=("baseline", "routing_geometry", "ablations"),
        default=["baseline"],
    )
    benchmark.add_argument("--output", type=Path, required=True, help="JSON measurement record")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "analyse-reconstruction":
            from moe_pilot.study_analysis import reconstruction_interactions

            print(
                json.dumps(
                    reconstruction_interactions(arguments.reconstruction, arguments.output),
                    indent=2,
                )
            )
            return 0
        if arguments.command == "report":
            from moe_pilot.reports import build_report

            print(build_report(arguments.reconstruction, arguments.study, arguments.output))
            return 0
        if arguments.command == "serve-report":
            from moe_pilot.reports import serve_report

            serve_report(arguments.directory, arguments.port, arguments.study)
            return 0
        if arguments.command == "download-models":
            from moe_pilot.acquisition import acquire_model
            from moe_pilot.catalog import load_catalog

            targets = {target.name: target for target in load_catalog(arguments.catalog)}
            if set(arguments.names) - set(targets):
                raise ValueError("Requested names are absent from the model catalog")
            for name in arguments.names:
                target = targets[name]
                print(f"Acquiring {name} at {target.revision}", flush=True)
                acquire_model(target.model, target.directory)
            return 0
        if arguments.command == "benchmark":
            from moe_pilot.benchmark import benchmark_engine
            from moe_pilot.resources import ResourcePolicy

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
                weight_cache=arguments.weight_cache,
                workloads=arguments.workloads,
                resources=ResourcePolicy(arguments.max_memory_gib, arguments.reserve_memory_gib),
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
        if arguments.command == "publish-reconstruction":
            from moe_pilot.publication import publish_reconstruction

            print(publish_reconstruction(config.paths.output, arguments.output))
            return 0
        if arguments.command == "download-study-data":
            from moe_pilot.study_data import acquire_sources

            acquire_sources(config)
            return 0
        if arguments.command == "inspect-study-data":
            from moe_pilot.study_data import inspect_sources

            report = inspect_sources(config, arguments.output)
            print(json.dumps(report, indent=2))
            return 0
        if arguments.command == "inventory":
            from moe_pilot.inventory import inspect_experiment

            print(inspect_experiment(config))
            return 0
        if arguments.command == "download-model":
            from moe_pilot.acquisition import acquire_model

            acquire_model(config.model, config.paths.model)
            return 0
        import torch

        torch.set_num_threads(config.execution.threads)
        if arguments.command == "study-analyse":
            from moe_pilot.study_analysis import analyse_study

            print(json.dumps(analyse_study(config), indent=2))
            return 0
        if arguments.command == "study-observe":
            from moe_pilot.study import collect

            collect(
                config,
                arguments.catalog,
                arguments.names,
                arguments.partition,
                max_batches=arguments.max_batches,
                interventions=arguments.interventions,
                formatting=arguments.formatting,
            )
            return 0
        if arguments.command == "study-select":
            from moe_pilot.study import freeze_selection

            print(json.dumps(freeze_selection(config), indent=2))
            return 0
        if arguments.command == "verify-model":
            from moe_pilot.artifacts import load_corpus
            from moe_pilot.engine import Engine
            from moe_pilot.verification import verify_local_model

            if arguments.documents < 1:
                raise ValueError("documents must be positive")
            if arguments.partition == "corpus":
                corpus = load_corpus(config.paths.data / "corpus.json")
            else:
                from moe_pilot.study_data import load_study_split

                corpus = load_study_split(config, arguments.partition)
            # First represent each available domain, then add remaining rows.
            chosen, seen = [], set()
            for index, record in enumerate(corpus):
                if record["domain"] not in seen:
                    chosen.append(index)
                    seen.add(record["domain"])
            chosen.extend(index for index in range(len(corpus)) if index not in chosen)
            engine = Engine(
                config.paths.model,
                device=config.execution.device,
                resources=config.execution.resource_policy(),
            )
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
                weight_cache=config.execution.weight_cache,
                resources=config.execution.resource_policy(),
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
