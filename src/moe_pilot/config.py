"""Validated experiment configuration; paths are relative to the TOML file."""

import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    model: Path
    data: Path
    output: Path

    def __post_init__(self):
        for name in ("model", "data", "output"):
            path = Path(getattr(self, name))
            if not path.is_absolute():
                raise ValueError(f"paths.{name} must be absolute in the library API")
            object.__setattr__(self, name, path.resolve())
        if self.output == self.data or self.output.is_relative_to(self.data):
            raise ValueError("Output must be separate from input data")
        if self.output == self.model or self.output.is_relative_to(self.model):
            raise ValueError("Output must be separate from model files")
        if self.data.is_relative_to(self.output) or self.model.is_relative_to(self.output):
            raise ValueError("Output cannot contain input data or model files")

    @property
    def artifacts(self):
        return self.output / "artifacts"

    @property
    def tables(self):
        return self.output / "tables"

    @property
    def figures(self):
        return self.output / "figures"

    @property
    def ablations(self):
        return self.output / "ablations"

    @property
    def routing(self):
        return self.artifacts / "routing.npz"

    @property
    def orthogonality(self):
        return self.artifacts / "orthogonality.npz"

    def prepare_outputs(self):
        for path in (self.artifacts, self.tables, self.figures, self.ablations):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Execution:
    threads: int = 8
    max_tokens: int = 64
    condition_group: int = 10


@dataclass(frozen=True)
class Analysis:
    seed: int = 0
    permutations: int = 20
    bootstrap_samples: int = 2000
    probe_tokens_per_domain: int = 100
    layers: tuple[int, ...] = ()

    def selected_layers(self, depth):
        layers = self.layers or tuple(dict.fromkeys((0, depth // 2, depth - 1)))
        if any(layer >= depth for layer in layers):
            raise ValueError(f"Analysis layers exceed model depth {depth}")
        return layers


@dataclass(frozen=True)
class Ablations:
    expert_counts: tuple[int, ...] = (1, 4, 8)
    control_seeds: int = 3
    control_candidates: int = 4000
    on_domain_candidates: int = 6000


@dataclass(frozen=True)
class Corpora:
    analysis_count: int = 100
    heldout_count: int = 40
    subtopic_count: int = 60


@dataclass(frozen=True)
class Model:
    repository: str = "allenai/OLMoE-1B-7B-0924"
    revision: str = ""


@dataclass(frozen=True)
class DatasetRevision:
    repository: str
    revision: str


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class Experiment:
    paths: Paths
    execution: Execution = field(default_factory=Execution)
    analysis: Analysis = field(default_factory=Analysis)
    ablations: Ablations = field(default_factory=Ablations)
    corpora: Corpora = field(default_factory=Corpora)
    model: Model = field(default_factory=Model)
    datasets: tuple[DatasetRevision, ...] = ()

    def __post_init__(self):
        for section in ("execution", "analysis", "ablations", "corpora"):
            for name, value in asdict(getattr(self, section)).items():
                if name in ("layers", "expert_counts"):
                    if not isinstance(value, tuple):
                        raise ValueError(f"{section}.{name} must contain unique integers")
                    for item in value:
                        _integer(item, f"{section}.{name}", 0 if name == "layers" else 1)
                    if len(value) != len(set(value)):
                        raise ValueError(f"{section}.{name} must contain unique integers")
                    if name == "expert_counts" and not value:
                        raise ValueError("ablations.expert_counts cannot be empty")
                else:
                    _integer(value, f"{section}.{name}", 0 if name == "seed" else 1)
        _integer(self.execution.max_tokens, "execution.max_tokens", 2)
        if not isinstance(self.model.repository, str) or not self.model.repository:
            raise ValueError("model.repository must be a nonempty string")
        if not isinstance(self.model.revision, str):
            raise ValueError("model.revision must be a string")
        if self.model.revision and not re.fullmatch(r"[0-9a-fA-F]{40}", self.model.revision):
            raise ValueError("model.revision must be a full 40-character commit SHA")
        if len({source.repository for source in self.datasets}) != len(self.datasets):
            raise ValueError("Dataset repositories must be unique")
        for source in self.datasets:
            if (
                not isinstance(source.repository, str)
                or not source.repository
                or not isinstance(source.revision, str)
                or not re.fullmatch(r"[0-9a-fA-F]{40}", source.revision)
            ):
                raise ValueError("Each dataset revision must be a full 40-character commit SHA")

    def record(self):
        record = asdict(self)
        record["paths"] = {name: str(value) for name, value in asdict(self.paths).items()}
        return record


def load_experiment(filename):
    """Resolve and validate all settings without creating directories or loading models."""
    filename = Path(filename).resolve()
    with filename.open("rb") as stream:
        raw = tomllib.load(stream)
    schema = raw.pop("schema_version", None)
    if type(schema) is not int or schema != 1:
        raise ValueError("schema_version must be 1")
    sections = {
        "execution": Execution,
        "analysis": Analysis,
        "ablations": Ablations,
        "corpora": Corpora,
        "model": Model,
    }
    unknown = set(raw) - {"paths", "datasets", *sections}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    paths = raw.pop("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("paths must be a TOML table")
    if set(paths) != {"model", "data", "output"}:
        raise ValueError("paths must define exactly model, data, and output")
    if any(not isinstance(value, str) or not value for value in paths.values()):
        raise ValueError("Configured paths must be nonempty strings")
    resolved = Paths(**{name: (filename.parent / value).resolve() for name, value in paths.items()})
    options = {}
    for name, constructor in sections.items():
        values = raw.get(name, {})
        if not isinstance(values, dict):
            raise ValueError(f"{name} must be a TOML table")
        values = dict(values)
        for key in ("layers", "expert_counts"):
            if key in values:
                if not isinstance(values[key], list):
                    raise ValueError(f"{name}.{key} must be an array")
                values[key] = tuple(values[key])
        try:
            options[name] = constructor(**values)
        except TypeError as error:
            raise ValueError(f"Invalid {name} settings: {error}") from error
    sources = raw.get("datasets", {})
    if not isinstance(sources, dict):
        raise ValueError("datasets must map repository names to commit SHAs")
    return Experiment(
        paths=resolved,
        datasets=tuple(
            DatasetRevision(repository, revision)
            for repository, revision in sorted(sources.items())
        ),
        **options,
    )
