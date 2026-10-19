import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from moe_pilot.model import ModelStore
from moe_pilot.smoke import tiny_model


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16, torch.float64])
def test_bounded_reader_matches_native_safetensors_exactly(tmp_path, dtype):
    tiny_model(tmp_path)
    path = tmp_path / "model.safetensors"
    with safe_open(path, framework="pt") as source:
        tensors = {key: source.get_tensor(key).to(dtype).clone() for key in source}
    save_file(tensors, path)
    store = ModelStore(tmp_path)
    with safe_open(path, framework="pt") as source:
        for name in source:
            actual = store.get(name)
            np.testing.assert_array_equal(actual.numpy(), source.get_tensor(name).float().numpy())
            assert store.tensor_bytes(name) == actual.numel() * 4
    assert not hasattr(store, "_handles")


def test_invalid_offsets_are_rejected_before_reading_values(tmp_path):
    tiny_model(tmp_path)
    store = ModelStore(tmp_path)
    name = next(iter(store.weight_map))
    store._headers[store.weight_map[name]][1][name]["data_offsets"] = [0, 1]
    with pytest.raises(ValueError, match="offsets"):
        store.get(name)
