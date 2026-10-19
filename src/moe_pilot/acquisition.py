"""Explicit immutable model acquisition and tensor-only safetensors conversion."""

import hashlib
import re
from pathlib import Path

from moe_pilot.artifacts import read_json, write_json


def _hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def convert_tensor_checkpoint(directory):
    """Convert indexed torch tensor shards without executing pickled model code.

    Source tensors retain their dtype and values. Original shards are preserved
    and identified separately from converted safetensors. No model is instantiated.
    """
    import torch
    from safetensors.torch import save_file

    directory = Path(directory)
    index = read_json(directory / "pytorch_model.bin.index.json")
    mapping = index["weight_map"]
    converted, manifest = {}, []
    for filename in sorted(set(mapping.values())):
        source = (directory / filename).resolve()
        if not source.is_relative_to(directory.resolve()) or source.suffix != ".bin":
            raise ValueError("Unsafe tensor shard path in model index")
        target = source.with_suffix(".safetensors")
        state = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
        expected = {name for name, shard in mapping.items() if shard == filename}
        if not isinstance(state, dict) or set(state) != expected:
            raise ValueError("Tensor shard keys differ from the checkpoint index")
        if any(not isinstance(tensor, torch.Tensor) for tensor in state.values()):
            raise ValueError("Checkpoint contains non-tensor state")
        tensors = {name: value.detach().contiguous().clone() for name, value in state.items()}
        temporary = target.with_suffix(".safetensors.partial")
        try:
            save_file(tensors, temporary, metadata={"format": "pt"})
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        relative = target.relative_to(directory.resolve()).as_posix()
        converted.update({name: relative for name in expected})
        manifest.append(
            {
                "source": filename,
                "source_sha256": _hash(source),
                "target": relative,
                "target_sha256": _hash(target),
                "tensors": len(tensors),
                "dtypes": sorted({str(t.dtype) for t in tensors.values()}),
            }
        )
        del state, tensors
    write_json(
        directory / "model.safetensors.index.json",
        {
            "metadata": index.get("metadata", {}),
            "weight_map": converted,
        },
    )
    return {
        "method": "torch.load(weights_only=True), tensor-preserving safetensors",
        "implementation_sha256": _hash(Path(__file__)),
        "shards": manifest,
    }


def acquire_model(model, directory):
    """Acquire an exact public snapshot; never overwrite a recorded model identity."""
    if not isinstance(model.revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", model.revision):
        raise ValueError("model.revision must be a full 40-character commit SHA")
    directory = Path(directory).resolve()
    marker = directory / "model_revision.json"
    if marker.exists():
        record = read_json(marker)
        if (record.get("repository"), record.get("revision")) != (model.repository, model.revision):
            raise ValueError("Model directory belongs to another immutable checkpoint")
        from moe_pilot.model import ModelStore

        store = ModelStore(directory)
        if any(not (directory / shard).is_file() for shard in set(store.weight_map.values())):
            raise ValueError("Recorded local model snapshot is incomplete")
        return directory
    from huggingface_hub import snapshot_download

    snapshot_download(
        model.repository,
        revision=model.revision,
        local_dir=directory,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.txt",
            "*.model",
            "pytorch_model*.bin",
            "README.md",
            "LICENSE*",
        ],
    )
    conversion = None
    if (
        not (directory / "model.safetensors").is_file()
        and not (directory / "model.safetensors.index.json").is_file()
    ):
        if not (directory / "pytorch_model.bin.index.json").is_file():
            raise ValueError("Model snapshot has no supported tensor checkpoint")
        conversion = convert_tensor_checkpoint(directory)
    from moe_pilot.model import ModelStore

    store = ModelStore(directory)
    shards = sorted(set(store.weight_map.values()))
    if any(not (directory / shard).is_file() for shard in shards):
        raise ValueError("Downloaded snapshot has missing tensor shards")
    write_json(
        marker,
        {
            "repository": model.repository,
            "revision": model.revision,
            "conversion": conversion,
            "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": _hash(path)}
                for path in sorted(directory.glob("*"))
                if path.is_file()
            },
        },
    )
    return directory
