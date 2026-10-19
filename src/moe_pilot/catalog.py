"""Pinned local model targets composed with the same experiment definition."""

import re
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from moe_pilot.config import Model, Paths


@dataclass(frozen=True)
class ModelTarget:
    name: str
    repository: str
    revision: str
    family: str
    lineage: str
    stage: str
    layers: tuple[int, ...]
    directory: Path

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.name):
            raise ValueError("Model target names must be safe identifiers")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", self.revision):
            raise ValueError("Model targets require immutable commit SHAs")
        if not all(
            isinstance(value, str) and value
            for value in (self.repository, self.family, self.lineage, self.stage)
        ):
            raise ValueError("Model targets require explicit family/lineage/stage metadata")
        if not self.layers or any(type(layer) is not int or layer < 0 for layer in self.layers):
            raise ValueError("Model target analysis layers must be nonnegative integers")
        if not self.directory.is_absolute():
            raise ValueError("Model target directories must be absolute")

    @property
    def model(self):
        return Model(self.repository, self.revision)

    def experiment(self, base):
        return replace(
            base,
            model=self.model,
            paths=Paths(self.directory, base.paths.data, base.paths.output / self.name),
            analysis=replace(base.analysis, layers=self.layers),
        )


def load_catalog(filename):
    filename = Path(filename).resolve()
    with filename.open("rb") as stream:
        raw = tomllib.load(stream)
    if set(raw) != {"schema_version", "root", "models"} or raw["schema_version"] != 1:
        raise ValueError("Model catalog requires schema_version, root, and models")
    root = (filename.parent / raw["root"]).resolve()
    targets = []
    for entry in raw["models"]:
        try:
            values = dict(entry)
            values["layers"] = tuple(values["layers"])
            targets.append(ModelTarget(directory=root / values["name"], **values))
        except (KeyError, TypeError) as error:
            raise ValueError(f"Invalid model catalog entry: {error}") from error
    if not targets or len({target.name for target in targets}) != len(targets):
        raise ValueError("Model catalog entries must be nonempty and unique")
    return tuple(targets)
