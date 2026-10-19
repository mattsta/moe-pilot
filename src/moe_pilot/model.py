"""Local model configuration and safetensors access; no implicit downloads."""

import hashlib
import json
from pathlib import Path

from safetensors import safe_open
from transformers import AutoConfig


def rope_theta(config):
    """Read the supported Transformers RoPE configuration."""
    parameters = config.rope_parameters
    if parameters.get("rope_type", "default") != "default":
        raise ValueError("Only default, unscaled RoPE is supported")
    return parameters["rope_theta"]


class ModelStore:
    """Read one tensor at a time from a local sharded or single-file checkpoint.

    Both the original per-expert checkpoint and Transformers' packed expert
    format are accepted. Tensors are materialized as CPU float32 values.
    """

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        if not (self.directory / "config.json").is_file():
            raise FileNotFoundError(f"Missing local model config: {self.directory / 'config.json'}")
        self.config = AutoConfig.from_pretrained(self.directory, local_files_only=True)
        self._handles = {}
        index = self.directory / "model.safetensors.index.json"
        if index.exists():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:
            weights = self.directory / "model.safetensors"
            if not weights.is_file():
                raise FileNotFoundError(f"No safetensors checkpoint in {self.directory}")
            with safe_open(weights, framework="pt") as handle:
                self.weight_map = dict.fromkeys(handle.keys(), weights.name)

    def get(self, name):
        if name not in self.weight_map:
            raise KeyError(f"Checkpoint has no tensor {name!r}")
        filename = self.weight_map[name]
        if filename not in self._handles:
            self._handles[filename] = safe_open(self.directory / filename, framework="pt")
        return self._handles[filename].get_tensor(name).float()

    def fingerprint(self):
        """Fingerprint config and shard identity without hashing multi-GB weights.

        Shards are identified by resolved path, size, and nanosecond mtime.
        This is a local resume guard, not a cryptographic model attestation.
        """
        shards = []
        for filename in sorted(set(self.weight_map.values())):
            path = self.directory / filename
            stat = path.stat()
            shards.append((str(path), stat.st_size, stat.st_mtime_ns))
        value = {"config": self.config.to_dict(), "weights": self.weight_map, "shards": shards}
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
