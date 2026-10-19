from dataclasses import replace

import pytest

from moe_pilot.config import Analysis, Execution, Paths, load_experiment


def test_paths_resolve_relative_to_config_file(config_file, tmp_path, monkeypatch):
    other = tmp_path / "unrelated"
    other.mkdir()
    monkeypatch.chdir(other)
    config = load_experiment(config_file)
    assert config.paths.data == tmp_path / "data"
    assert config.paths.output == tmp_path / "output"
    assert not config.paths.output.exists()


@pytest.mark.parametrize(
    "old,new",
    [
        ("threads = 2", "threads = 0"),
        ("max_tokens = 32", "max_tokens = 1"),
        ("seed = 17", "seed = true"),
        ("layers = [0, 1]", "layers = [0, 0]"),
        ("expert_counts = [1]", "expert_counts = []"),
        ("permutations = 3", "permutatons = 3"),
        ("schema_version = 1", "schema_version = 2"),
    ],
)
def test_invalid_settings_are_rejected(config_file, old, new):
    config_file.write_text(config_file.read_text().replace(old, new))
    with pytest.raises(ValueError):
        load_experiment(config_file)


def test_data_model_and_outputs_are_separate(tmp_path):
    with pytest.raises(ValueError):
        Paths(tmp_path / "model", tmp_path / "data", tmp_path / "data/output")


def test_library_configuration_is_immutable_and_validated(experiment):
    with pytest.raises(ValueError):
        replace(experiment, execution=Execution(max_tokens=1))
    with pytest.raises(ValueError):
        Analysis(layers=(8,)).selected_layers(2)
