"""Thin CLI over the same explicit experiment API used by library callers."""

import argparse
import re
import sys
from pathlib import Path

from moe_pilot.config import load_experiment
from moe_pilot.experiment import STEPS, ExperimentRunner


def parser():
    result = argparse.ArgumentParser(description="Configured CPU MoE research experiments")
    result.add_argument(
        "--config", type=Path, help="Experiment TOML file; required except for smoke"
    )
    commands = result.add_subparsers(dest="command", required=True)
    for name in STEPS:
        commands.add_parser(name, help=f"Run the {name} experiment step")
    commands.add_parser("run", help="Run inference, conditions, ablations, and reports in order")
    commands.add_parser("download-model", help="Download the exact configured model revision")
    commands.add_parser("inventory", help="Read configured input and output availability")
    smoke = commands.add_parser("smoke", help="Offline tiny-model numerical verification")
    smoke.add_argument("--output", type=Path, help="Optional JSON report destination")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "smoke":
            from moe_pilot.smoke import verify_engine

            report = verify_engine(arguments.output)
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
        runner = ExperimentRunner(config)
        if arguments.command == "run":
            runner.run_pipeline()
        else:
            runner.run(arguments.command)
    except (OSError, ValueError) as error:
        print(f"moe-pilot: {error}", file=sys.stderr)
        return 1
    return 0
