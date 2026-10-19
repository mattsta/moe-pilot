import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from filelock import FileLock

from moe_pilot.artifacts import read_json, write_json
from moe_pilot.experiment import ExperimentRunner


def install_stage(monkeypatch, module, callback):
    monkeypatch.setitem(sys.modules, f"moe_pilot.workflows.{module}", SimpleNamespace(run=callback))


def test_completed_outputs_are_reused_and_modified_inputs_are_recomputed(experiment, monkeypatch):
    calls = []
    write_json(experiment.paths.data / "corpus.json", [{"text": "first"}])

    def routing(config):
        calls.append(1)
        config.paths.routing.write_text(str(len(calls)))
        return [config.paths.routing]

    install_stage(monkeypatch, "routing", routing)
    runner = ExperimentRunner(experiment)
    runner.run("routing")
    runner.run("routing")
    assert len(calls) == 1
    write_json(experiment.paths.data / "corpus.json", [{"text": "changed"}])
    runner.run("routing")
    assert len(calls) == 2


def test_tampered_dependency_is_rejected_before_downstream_execution(experiment, monkeypatch):
    def routing(config):
        config.paths.routing.write_text("original")
        return [config.paths.routing]

    install_stage(monkeypatch, "routing", routing)
    runner = ExperimentRunner(experiment)
    runner.run("routing")
    experiment.paths.routing.write_text("tampered")
    with pytest.raises(ValueError, match="routing.*stale"):
        runner.run("domain-report")


def test_changed_config_cannot_reuse_output_directory(experiment, monkeypatch):
    def routing(config):
        config.paths.routing.write_text("routing")
        return [config.paths.routing]

    install_stage(monkeypatch, "routing", routing)
    ExperimentRunner(experiment).run("routing")
    changed = replace(experiment, analysis=replace(experiment.analysis, seed=99))
    with pytest.raises(ValueError, match="configuration changed"):
        ExperimentRunner(changed).run("routing")


def test_failed_stages_are_recorded_and_never_consumed(experiment, monkeypatch):
    def failed(config):
        config.paths.routing.write_text("partial")
        raise RuntimeError("interrupted")

    install_stage(monkeypatch, "routing", failed)
    runner = ExperimentRunner(experiment)
    with pytest.raises(RuntimeError, match="interrupted"):
        runner.run("routing")
    record = read_json(experiment.paths.output / "experiment.json")
    assert record["steps"]["routing"]["status"] == "failed"
    with pytest.raises(ValueError, match="routing"):
        runner.run("domain-report")


def test_single_writer_lock_blocks_concurrent_mutation(experiment):
    experiment.paths.output.mkdir()
    with FileLock(experiment.paths.output / ".experiment.lock"):
        with pytest.raises(ValueError, match="Another process"):
            ExperimentRunner(experiment).run("routing")


def test_inputs_changed_during_execution_are_not_accepted(experiment, monkeypatch):
    write_json(experiment.paths.data / "corpus.json", [{"text": "before"}])

    def changing_inputs(config):
        config.paths.routing.write_text("partial")
        write_json(config.paths.data / "corpus.json", [{"text": "after"}])
        return [config.paths.routing]

    install_stage(monkeypatch, "routing", changing_inputs)
    with pytest.raises(ValueError, match="Inputs changed during"):
        ExperimentRunner(experiment).run("routing")


def test_inventory_does_not_mutate_an_existing_manifest(experiment, monkeypatch):
    from moe_pilot.inventory import inspect_experiment

    experiment.paths.output.mkdir()
    filename = experiment.paths.output / "experiment.json"
    write_json(filename, {"steps": {"routing": {"status": "failed"}}})
    before = filename.read_bytes()
    assert "routing: failed" in inspect_experiment(experiment)
    assert filename.read_bytes() == before
