"""One experiment lifecycle for the CLI and library, with artifact provenance."""

import hashlib
import importlib
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from filelock import FileLock, Timeout

from moe_pilot.artifacts import read_json, write_json
from moe_pilot.config import Experiment


@dataclass(frozen=True)
class Step:
    module: str
    function: str
    dependencies: tuple[str, ...] = ()
    data_files: tuple[str, ...] = ()
    model_inputs: bool = True
    study_sources: bool = False


STEPS = {
    "build-corpus": Step("build_corpus", "run", model_inputs=False),
    "build-heldout": Step("build_heldout", "run", data_files=("corpus.json",), model_inputs=False),
    "build-subtopics": Step("build_subtopics", "run", model_inputs=False),
    "routing": Step("routing", "run", data_files=("corpus.json", "corpus_sources.json")),
    "domain-report": Step(
        "domain_report", "run", ("routing",), ("corpus.json", "corpus_sources.json")
    ),
    "build-conditions": Step(
        "conditions", "run", ("domain-report",), ("corpus.json", "corpus_sources.json")
    ),
    "ablate": Step(
        "ablations",
        "run",
        ("build-conditions",),
        ("corpus.json", "heldout.json", "heldout_sources.json"),
    ),
    "ablation-report": Step(
        "ablation_report", "run", ("ablate",), ("heldout.json", "heldout_sources.json")
    ),
    "subtopic-report": Step(
        "subtopic_report",
        "run",
        ("domain-report",),
        ("subtopic_corpus.json", "subtopic_sources.json"),
    ),
    "prepare-study": Step("study_data", "run", model_inputs=False, study_sources=True),
}
PIPELINE = (
    "routing",
    "domain-report",
    "build-conditions",
    "ablate",
    "ablation-report",
    "subtopic-report",
)


def _hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _files(paths):
    return {str(path): _hash(path) for path in paths if path.is_file()}


def _versions(selection):
    import torch

    versions = {
        name: version(name)
        for name in (
            "moe-pilot",
            "torch",
            "transformers",
            "numpy",
            "pandas",
            "matplotlib",
            "safetensors",
            "huggingface-hub",
            "filelock",
        )
    }
    versions.update(
        device=selection.identity(),
        machine=platform.machine(),
        torch_threads=torch.get_num_threads(),
        torch_matmul_precision=torch.get_float32_matmul_precision(),
    )
    return versions


class ExperimentRunner:
    """Execute a configured step without changing directories or process settings.

    Each output directory belongs to one immutable configuration. Completed
    steps record exact inputs and output hashes; dependencies are checked
    recursively before downstream results can be consumed. Single writer.
    """

    def __init__(self, experiment: Experiment):
        self.config = experiment
        self.manifest = experiment.paths.output / "experiment.json"
        self._selection = None

    @property
    def selection(self):
        from moe_pilot.devices import resolve_device

        if self._selection is None:
            self._selection = resolve_device(self.config.execution.device)
        return self._selection

    def _record(self):
        if not self.manifest.exists():
            return {"schema": 1, "configuration": self.config.record(), "steps": {}}
        record = read_json(self.manifest)
        # JSON round-trip normalizes tuple settings to arrays.
        import json

        if record.get("schema") != 1 or record.get("configuration") != json.loads(
            json.dumps(self.config.record())
        ):
            raise ValueError(
                "Experiment configuration changed; select a new paths.output directory"
            )
        return record

    def _inputs(self, step):
        paths = self.config.paths
        package = Path(__file__).parent
        inputs = _files(sorted(package.rglob("*.py")))
        specification = STEPS[step]
        inputs.update(_files([paths.data / name for name in specification.data_files]))
        if specification.study_sources:
            from moe_pilot.study_data import source_record_path

            inputs.update(
                _files(
                    [
                        path
                        for source in self.config.study.sources
                        for path in (source.path, source_record_path(source))
                    ]
                )
            )
        if specification.model_inputs:
            inputs.update(_files(sorted(paths.model.glob("*.json"))))
            for shard in sorted(paths.model.glob("*.safetensors")):
                stat = shard.stat()
                inputs[str(shard)] = f"{stat.st_size}:{stat.st_mtime_ns}"
        return inputs

    def _valid(self, step, records):
        record = records.get(step, {})
        if record.get("versions") != _versions(self.selection):
            return False
        if record.get("status") != "complete" or record.get("inputs") != self._inputs(step):
            return False
        if any(not self._valid(parent, records) for parent in STEPS[step].dependencies):
            return False
        if any(
            not Path(path).is_file() or _hash(Path(path)) != digest
            for path, digest in record.get("outputs", {}).items()
        ):
            return False
        return record.get("dependencies") == {
            parent: records[parent]["outputs"] for parent in STEPS[step].dependencies
        }

    def run(self, step):
        if step not in STEPS:
            raise ValueError(f"Unknown experiment step: {step}")
        self.config.paths.output.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(self.config.paths.output / ".experiment.lock", timeout=0):
                self._run_step(step)
        except Timeout as error:
            raise ValueError("Another process is writing this experiment") from error

    def _run_step(self, step):
        if step not in STEPS:
            raise ValueError(f"Unknown experiment step: {step}")
        record = self._record()
        for parent in STEPS[step].dependencies:
            if not self._valid(parent, record["steps"]):
                raise ValueError(
                    f"Run {parent!r} first: its inputs or outputs are missing or stale"
                )
        if self._valid(step, record["steps"]):
            return
        self.config.paths.prepare_outputs()
        current = {
            "status": "running",
            "inputs": self._inputs(step),
            "dependencies": {
                parent: record["steps"][parent]["outputs"] for parent in STEPS[step].dependencies
            },
            "versions": _versions(self.selection),
            "device_selection": self.selection.record(),
        }
        record["steps"][step] = current
        write_json(self.manifest, record)
        try:
            specification = STEPS[step]
            module = importlib.import_module(f"moe_pilot.workflows.{specification.module}")
            outputs = getattr(module, specification.function)(self.config, device=self.selection)
            if not outputs:
                raise ValueError(f"Step {step} must declare its output artifacts")
            missing = [str(path) for path in outputs if not Path(path).is_file()]
            if missing:
                raise FileNotFoundError(f"Step {step} did not produce {missing}")
            if current["inputs"] != self._inputs(step):
                raise ValueError(f"Inputs changed during {step}; results were not accepted")
            current.update(status="complete", outputs=_files(outputs))
        except BaseException as error:
            current.update(status="failed", error=str(error))
            raise
        finally:
            write_json(self.manifest, record)

    def run_pipeline(self):
        for step in PIPELINE:
            self.run(step)
