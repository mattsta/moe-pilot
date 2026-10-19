"""Atomic experiment checkpoints with explicit input and condition identities."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def atomic_array(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class Checkpoints:
    """A single-writer checkpoint directory; separate runs need separate directories."""

    def __init__(self, directory, identity, conditions):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest = self.directory / "manifest.json"
        self.data = {"schema": 1, "identity": identity, "conditions": {}}
        if self.manifest.exists():
            self.data = json.loads(self.manifest.read_text())
            if self.data.get("schema") != 1 or self.data.get("identity") != identity:
                raise ValueError("Checkpoint inputs changed; use a new output directory")
        elif any(self.directory.glob("losses/*.npy")) or any(self.directory.glob("cache/*.npy")):
            raise ValueError(
                "Untracked checkpoint files lack provenance; use a new output directory"
            )
        for name, mask in conditions.items():
            signature = hashlib.sha256(json.dumps(mask, sort_keys=True).encode()).hexdigest()
            previous = self.data["conditions"].get(name)
            if previous is not None and previous != signature:
                raise ValueError(
                    f"Checkpoint condition {name!r} changed; use a new output directory"
                )
            self.data["conditions"][name] = signature
        descriptor, filename = tempfile.mkstemp(dir=self.directory, suffix=".json")
        temporary = Path(filename)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(self.data, stream, indent=2, sort_keys=True)
                stream.write("\n")
            temporary.replace(self.manifest)
        finally:
            temporary.unlink(missing_ok=True)

    def path(self, relative):
        return self.directory / relative

    def exists(self, relative):
        return self.path(relative).is_file()

    def save(self, relative, array):
        atomic_array(self.path(relative), array)

    def load(self, relative):
        return np.load(self.path(relative), allow_pickle=False)
