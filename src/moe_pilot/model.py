"""Local model configuration and safetensors access; no implicit downloads."""

import hashlib
import json
import math
from pathlib import Path

import torch
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
    format are accepted. Tensors are staged as host float32 values; the engine
    transfers each needed tensor to its selected compute device.
    """

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        if not (self.directory / "config.json").is_file():
            raise FileNotFoundError(f"Missing local model config: {self.directory / 'config.json'}")
        self.config = AutoConfig.from_pretrained(self.directory, local_files_only=True)
        self._headers = {}
        index = self.directory / "model.safetensors.index.json"
        if index.exists():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:
            weights = self.directory / "model.safetensors"
            if not weights.is_file():
                raise FileNotFoundError(f"No safetensors checkpoint in {self.directory}")
            with safe_open(weights, framework="pt") as handle:
                self.weight_map = dict.fromkeys(handle.keys(), weights.name)
        for filename in set(self.weight_map.values()):
            path = (self.directory / filename).resolve()
            if not path.is_relative_to(self.directory):
                raise ValueError("Weight shards must remain inside the model directory")
            with path.open("rb") as stream:
                size = int.from_bytes(stream.read(8), "little")
                if not 2 <= size <= min(100_000_000, path.stat().st_size - 8):
                    raise ValueError("Invalid safetensors header size")
                header = json.loads(stream.read(size))
            self._headers[filename] = (size + 8, header)

    def get(self, name):
        if name not in self.weight_map:
            raise KeyError(f"Checkpoint has no tensor {name!r}")
        filename = self.weight_map[name]
        start, header = self._headers[filename]
        entry = header[name]
        dtypes = {
            "F32": torch.float32,
            "BF16": torch.bfloat16,
            "F16": torch.float16,
            "F64": torch.float64,
        }
        if entry["dtype"] not in dtypes:
            raise ValueError("Model weights must use a supported floating-point dtype")
        dtype = dtypes[entry["dtype"]]
        shape = entry["shape"]
        if not isinstance(shape, list) or any(
            type(value) is not int or value < 1 for value in shape
        ):
            raise ValueError("Invalid model tensor shape")
        first, last = entry["data_offsets"]
        count = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        path = self.directory / filename
        if first < 0 or last - first != count or start + last > path.stat().st_size:
            raise ValueError("Invalid safetensors tensor offsets")
        # Parse shard metadata once, then read only this tensor into an owned
        # host buffer. Repeated safe_open calls reparse thousands of entries;
        # retained mappings accumulate the whole checkpoint's resident pages.
        buffer = bytearray(count)
        view = memoryview(buffer)
        with path.open("rb") as stream:
            stream.seek(start + first)
            offset = 0
            while offset < count:
                read = stream.readinto(view[offset:])
                if not read:
                    raise ValueError("Truncated model tensor")
                offset += read
        return torch.frombuffer(buffer, dtype=dtype).reshape(shape).float()

    def tensor_bytes(self, name):
        """Float32 allocation size from safetensors metadata, without loading values."""
        return math.prod(self._headers[self.weight_map[name]][1][name]["shape"]) * 4

    def embedding_rows(self, token_ids):
        """Convert only requested rows, rather than the full embedding table."""
        filename = self.weight_map["model.embed_tokens.weight"]
        with safe_open(self.directory / filename, framework="pt") as handle:
            return handle.get_tensor("model.embed_tokens.weight")[token_ids].float()

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
        value = {
            "config": self.config.to_dict(),
            "config_sha256": hashlib.sha256(
                (self.directory / "config.json").read_bytes()
            ).hexdigest(),
            "weights": self.weight_map,
            "shards": shards,
        }
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
