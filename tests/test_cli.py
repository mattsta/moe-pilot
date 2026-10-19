import importlib
from pathlib import Path

import pytest

from moe_pilot.cli import WORKFLOWS, main


def test_help_does_not_need_a_model(capsys):
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    assert "build-conditions" in capsys.readouterr().out


@pytest.mark.parametrize("module", [spec[0] for spec in WORKFLOWS.values()])
def test_workflow_imports_do_not_execute(module, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    imported = importlib.import_module(f"moe_pilot.workflows.{module}")
    assert callable(imported.main)
    assert not list(tmp_path.iterdir())


def test_inventory_without_weights(tmp_path, capsys):
    assert main(["--workspace", str(tmp_path), "inventory"]) == 0
    assert "Model config: missing" in capsys.readouterr().out


def test_model_download_requires_an_immutable_revision(tmp_path, capsys):
    with pytest.raises(SystemExit) as result:
        main(["--workspace", str(tmp_path), "download-model", "--revision", "main"])
    assert result.value.code == 2
    assert "40-character" in capsys.readouterr().err


def test_cli_restores_directory_after_failure(tmp_path, capsys):
    previous = Path.cwd()
    assert main(["--workspace", str(tmp_path), "forward"]) == 1
    assert Path.cwd() == previous
    assert "Missing local model config" in capsys.readouterr().out
