from dataclasses import replace

import numpy as np
import pytest
import torch

from moe_pilot.acquisition import acquire_model, convert_tensor_checkpoint
from moe_pilot.artifacts import write_json
from moe_pilot.catalog import load_catalog
from moe_pilot.config import Model
from moe_pilot.engine import Engine


def test_tensor_only_conversion_preserves_full_engine_outputs(local_model, tmp_path):
    reference, original = local_model
    directory = tmp_path / "converted"
    reference.config.save_pretrained(directory)
    state = reference.state_dict()
    source = directory / "pytorch_model-00001-of-00001.bin"
    torch.save(state, source)
    write_json(
        directory / "pytorch_model.bin.index.json",
        {
            "metadata": {},
            "weight_map": dict.fromkeys(state, source.name),
        },
    )
    record = convert_tensor_checkpoint(directory)
    assert source.is_file()
    assert record["shards"][0]["source_sha256"] != record["shards"][0]["target_sha256"]
    converted = Engine(directory, device="cpu")
    batch = original.batch([[2, 3, 4], [5, 6]], 4)
    first, _, _ = original.run(batch, {}, verbose=False)
    second, _, _ = converted.run(batch, {}, verbose=False)
    np.testing.assert_array_equal(first["base"], second["base"])
    assert not list(directory.glob("*.partial"))


def test_conversion_rejects_non_tensor_state(tmp_path):
    torch.save({"bad": "not a tensor"}, tmp_path / "weights.bin")
    write_json(tmp_path / "pytorch_model.bin.index.json", {"weight_map": {"bad": "weights.bin"}})
    with pytest.raises(ValueError, match="non-tensor"):
        convert_tensor_checkpoint(tmp_path)
    assert not (tmp_path / "model.safetensors.index.json").exists()


def test_existing_model_identity_is_not_rewritten_or_downloaded(local_model, monkeypatch):
    _, engine = local_model
    marker = engine.store.directory / "model_revision.json"
    model = Model("example/local", "a" * 40)
    write_json(marker, {"repository": model.repository, "revision": model.revision})
    before = marker.read_bytes()
    import huggingface_hub

    def forbidden(*args, **kwargs):
        pytest.fail("An intact recorded snapshot must not be downloaded again")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden)
    assert acquire_model(model, engine.store.directory) == engine.store.directory
    assert marker.read_bytes() == before
    with pytest.raises(ValueError, match="another immutable"):
        acquire_model(replace(model, revision="b" * 40), engine.store.directory)


def test_catalog_composes_the_same_experiment_api_and_resolves_paths(experiment, tmp_path):
    filename = tmp_path / "catalog.toml"
    filename.write_text("""schema_version = 1
root = "models"
[[models]]
name = "local-model"
repository = "example/local"
revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
family = "olmoe"
lineage = "test-lineage"
stage = "base"
layers = [1]
""")
    (target,) = load_catalog(filename)
    configured = target.experiment(experiment)
    assert configured.paths.model == tmp_path / "models/local-model"
    assert configured.paths.output == experiment.paths.output / "local-model"
    assert configured.study == experiment.study
    assert configured.analysis.layers == (1,)
    assert configured.model == target.model
