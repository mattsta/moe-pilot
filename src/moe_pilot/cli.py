"""Documented command-line interface for research workspaces."""

import argparse
import importlib
import re
from contextlib import chdir
from pathlib import Path

WORKFLOWS = {
    "build-corpus": ("build_corpus", "Download and sample the five-domain corpus"),
    "build-heldout": ("build_heldout", "Build held-out texts disjoint from corpus.json"),
    "build-subtopics": ("build_subtopics", "Download and sample subtopic corpora"),
    "forward": ("forward", "Stream the model and save routing/orthogonality arrays"),
    "analyze": ("analyze", "Analyze routing and orthogonality; write CSVs and plots"),
    "activation-extra": ("act_extra", "Compute additional activation statistics"),
    "build-conditions": ("ablate", "Build specialist and matched-control masks"),
    "ablate": ("ablate_run", "Run checkpointed held-out ablations"),
    "analyze-ablations": ("ablate_analyze", "Summarize ablations and expand condition aliases"),
    "ondomain-controls": ("ablate_ondomain_ctrl", "Run on-domain mass-matched controls"),
    "subtopics": ("subtopics", "Run and analyze subtopic routing"),
}


def parser():
    result = argparse.ArgumentParser(description="OLMoE CPU research workflows")
    result.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Corpus and output directory (default: current directory)",
    )
    result.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/olmoe"),
        help="Local model directory, relative to workspace unless absolute",
    )
    result.add_argument("--threads", type=int, default=8, help="CPU threads (default: 8)")
    commands = result.add_subparsers(dest="command", required=True)
    for name, (_, description) in WORKFLOWS.items():
        command = commands.add_parser(name, help=description, description=description)
        if name == "forward":
            command.add_argument("max_tokens", type=int, nargs="?", default=64)
            command.add_argument("layers", type=int, nargs="?", default=None)
        if name in ("build-conditions", "ondomain-controls"):
            command.add_argument(
                "--control-candidates",
                type=int,
                default=4000 if name == "build-conditions" else 6000,
                help="Random subsets searched per multi-expert control",
            )
    download = commands.add_parser("download-model", help="Explicitly download a model snapshot")
    download.add_argument("--revision", required=True, help="Model commit SHA for reproducibility")
    commands.add_parser("smoke", help="Offline tiny-model numerical and checkpoint verification")
    commands.add_parser("inventory", help="Inspect available inputs and historical result tables")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    if arguments.threads < 1:
        parser().error("--threads must be positive")
    workspace = arguments.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    model_dir = arguments.model_dir
    if not model_dir.is_absolute():
        model_dir = workspace / model_dir
    if arguments.command == "inventory":
        from moe_pilot.inventory import inspect_workspace

        print(inspect_workspace(workspace, model_dir))
        return 0
    if arguments.command == "download-model":
        if not re.fullmatch(r"[0-9a-fA-F]{40}", arguments.revision):
            parser().error("--revision must be a full 40-character model commit SHA")
        from huggingface_hub import snapshot_download

        from moe_pilot.artifacts import write_json

        snapshot_download(
            "allenai/OLMoE-1B-7B-0924",
            revision=arguments.revision,
            local_dir=model_dir,
            allow_patterns=["*.json", "*.safetensors", "*.txt"],
        )
        write_json(
            model_dir / "model_revision.json",
            {
                "repository": "allenai/OLMoE-1B-7B-0924",
                "revision": arguments.revision,
            },
        )
        return 0
    import torch

    torch.set_num_threads(arguments.threads)
    try:
        if arguments.command == "smoke":
            from moe_pilot.smoke import verify_engine

            report = verify_engine(workspace / "results" / "smoke.json")
            print(f"Offline smoke passed; max loss error {report['max_loss_error']:.3g}")
            return 0
        module_name = WORKFLOWS[arguments.command][0]
        module = importlib.import_module(f"moe_pilot.workflows.{module_name}")
        options = {"model_dir": model_dir}
        if arguments.command in ("build-conditions", "ondomain-controls"):
            if arguments.control_candidates < 1:
                raise ValueError("--control-candidates must be positive")
            options["control_candidates"] = arguments.control_candidates
        if arguments.command == "forward":
            if arguments.max_tokens < 2 or (arguments.layers is not None and arguments.layers < 1):
                raise ValueError("max_tokens must be >=2; layers must be positive")
            options.update(max_tokens=arguments.max_tokens, layers=arguments.layers)
        # Research file schemas are workspace-relative. Only the CLI changes
        # directory, and it restores the caller's directory on exit.
        with chdir(workspace):
            module.main(**options)
    except (FileNotFoundError, ValueError) as error:
        print(f"moe-pilot: {error}")
        print("See docs/runbook.md for prerequisites and pipeline order.")
        return 1
    return 0
