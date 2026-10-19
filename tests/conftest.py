import os

# Tests must never fetch a model or corpus, even if a dependency tries to.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["MPLBACKEND"] = "Agg"

import pytest
import torch

from moe_pilot.config import load_experiment
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


@pytest.fixture
def config_file(tmp_path):
    directory = tmp_path / "configuration"
    directory.mkdir()
    filename = directory / "experiment.toml"
    filename.write_text("""schema_version = 1
[paths]
model = "../model"
data = "../data"
output = "../output"
[execution]
threads = 2
max_tokens = 32
condition_group = 4
[analysis]
seed = 17
layers = [0, 1]
permutations = 3
bootstrap_samples = 20
probe_tokens_per_domain = 10
[ablations]
expert_counts = [1]
control_seeds = 1
control_candidates = 8
on_domain_candidates = 8
""")
    return filename


@pytest.fixture
def experiment(config_file):
    return load_experiment(config_file)
