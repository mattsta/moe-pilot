"""Local model configuration and safetensors access; no implicit downloads."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig


def rope_theta(config):
    """Read the unscaled RoPE base from both Transformers config formats."""
    parameters = getattr(config, "rope_parameters", None)
    if parameters:
        if parameters.get("rope_type", "default") != "default":
            raise ValueError("Only default, unscaled RoPE is supported")
        return parameters["rope_theta"]
    if getattr(config, "rope_scaling", None):
        raise ValueError("Only default, unscaled RoPE is supported")
    return config.rope_theta


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
            # Transformers 5 packs expert parameters. Expose the original names
            # to keep the reference forward and activation probes compatible.
            prefix, separator, suffix = name.partition(".mlp.experts.")
            if separator:
                expert, projection, _ = suffix.split(".")
                base = prefix + ".mlp.experts."
                if projection == "down_proj":
                    return self.get(base + "down_proj")[int(expert)]
                packed = self.get(base + "gate_up_proj")[int(expert)]
                return packed.chunk(2, dim=0)[projection == "up_proj"]
            raise KeyError(f"Checkpoint has no tensor {name!r}")
        filename = self.weight_map[name]
        if filename not in self._handles:
            self._handles[filename] = safe_open(self.directory / filename, framework="pt")
        return self._handles[filename].get_tensor(name).float()

    def expert_weights(self, layer, projection):
        """Materialize a projection across experts without repeated packed reads."""
        prefix = f"model.layers.{layer}.mlp.experts."
        if prefix + "gate_up_proj" in self.weight_map:
            if projection == "down":
                return self.get(prefix + "down_proj")
            return self.get(prefix + "gate_up_proj").chunk(2, dim=1)[projection == "up"]
        return torch.stack(
            [
                self.get(prefix + f"{expert}.{projection}_proj.weight")
                for expert in range(self.config.num_experts)
            ]
        )

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
