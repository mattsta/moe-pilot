import os

# Tests must never fetch a model or corpus, even if a dependency tries to.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["MPLBACKEND"] = "Agg"

import pytest
import torch

from moe_pilot.engine import Engine
from moe_pilot.smoke import tiny_model


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def local_model(tmp_path):
    directory = tmp_path / "model"
    reference = tiny_model(directory)
    return reference, Engine(directory)
