import pytest

from moe_pilot.config import load_experiment
from moe_pilot.datasets import DatasetLoader


def test_dataset_download_requires_a_pinned_source(experiment):
    with pytest.raises(ValueError, match="immutable commit SHA"):
        DatasetLoader(experiment).load("some/dataset", split="test")


def test_config_rejects_mutable_dataset_revision(config_file):
    config_file.write_text(config_file.read_text() + '\n[datasets]\n"some/dataset" = "main"\n')
    with pytest.raises(ValueError, match="40-character"):
        load_experiment(config_file)
