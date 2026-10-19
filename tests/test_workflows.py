"""Offline integration of research artifact production and corpus orchestration."""

import json
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from moe_pilot.artifacts import read_json, write_json
from moe_pilot.cli import main
from moe_pilot.config import DatasetRevision, load_experiment
from moe_pilot.experiment import ExperimentRunner
from moe_pilot.interventions import Conditions
from moe_pilot.smoke import tiny_model


@pytest.fixture
def research_experiment(config_file):
    config = load_experiment(config_file)
    model_dir = config.paths.model
    tiny_model(model_dir)
    vocabulary = {"[UNK]": 0, **{f"token{i}": i for i in range(1, 32)}}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(
        model_dir
    )
    domains = ["code", "math", "multilingual", "prose", "science"]
    corpus = [
        {"domain": domain, "text": " ".join(f"token{(i + offset) % 31 + 1}" for i in range(48))}
        for domain in domains
        for offset in range(4)
    ]
    write_json(config.paths.data / "corpus.json", corpus)
    write_json(
        config.paths.data / "heldout.json",
        [{**record, "text": "token31 " + record["text"] + " heldout"} for record in corpus],
    )
    write_json(
        config.paths.data / "subtopic_corpus.json",
        [
            {**document, "subtopic": f"topic{index % 2}"}
            for index, document in enumerate(corpus)
            if document["domain"] != "prose"
        ],
    )
    return config


@pytest.mark.parametrize("device_policy", ["cpu", "auto"])
def test_complete_configured_pipeline_from_an_unrelated_directory(
    research_experiment, config_file, tmp_path, monkeypatch, device_policy
):
    config_file.write_text(
        config_file.read_text().replace('device = "cpu"', f'device = "{device_policy}"')
    )
    config = load_experiment(config_file)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    with patch("os.chdir", side_effect=AssertionError("Directory change forbidden")):
        assert main(["--config", str(config_file), "run"]) == 0
    assert not list(unrelated.iterdir())
    paths = config.paths
    with np.load(paths.routing) as routing:
        assert routing["topi"].shape == (2, 640, 2)
        assert np.isfinite(routing["loss"]).all()
    assert len(pd.read_csv(paths.tables / "layer_domain_metrics.csv")) == 10
    assert len(pd.read_csv(paths.tables / "activation_orthogonality.csv")) == 2
    assert (paths.figures / "jsd_by_layer.png").stat().st_size > 1000
    conditions = Conditions.load(paths.ablations / "conditions.json")
    assert len(conditions.items) == 45
    assert {item.family for item in conditions.items} == {
        "specialist",
        "global-control",
        "on-domain-control",
    }
    assert len(pd.read_csv(paths.tables / "ablations/delta_loss.csv")) == 225
    assert len(pd.read_csv(paths.tables / "ablations/ablation_summary.csv")) == 30
    assert len(pd.read_csv(paths.tables / "subtopics/layer_within_vs_between_jsd.csv")) == 8
    assert (paths.figures / "subtopics/jsd_within_vs_between.png").stat().st_size > 1000
    manifest = read_json(paths.output / "experiment.json")
    assert all(record["status"] == "complete" for record in manifest["steps"].values())
    assert manifest["steps"]["routing"]["device_selection"]["requested"] == device_policy
    # A second pipeline call should reuse only fully validated artifacts.
    before = paths.routing.stat().st_mtime_ns
    ExperimentRunner(config).run_pipeline()
    assert paths.routing.stat().st_mtime_ns == before


def test_publication_validates_completed_artifacts_and_exports_traceable_results(
    research_experiment, tmp_path
):
    from moe_pilot.publication import publish_reconstruction, validate_evidence

    config = research_experiment
    ExperimentRunner(config).run_pipeline()
    destination = tmp_path / "publication"
    summary = publish_reconstruction(config.paths.output, destination)
    evidence = read_json(destination / "evidence.json")
    assert evidence["derived"]["documents"] == 20
    assert evidence["derived"]["conditions"] == 45
    assert "exploratory" in summary.read_text()
    assert "permutations" in (destination / "analysis.md").read_text()
    assert (destination / "figures/jsd_by_layer.png").is_file()
    assert not (destination / "artifacts/routing.npz").exists()
    assert all("sha256" in artifact for artifact in evidence["artifacts"])
    table = config.paths.tables / "ablations/ablation_summary.csv"
    original = table.read_bytes()
    table.write_bytes(original + b"tampered")
    with pytest.raises(ValueError, match="output"):
        publish_reconstruction(config.paths.output, tmp_path / "invalid")
    table.write_bytes(original)
    manifest_path = config.paths.output / "experiment.json"
    manifest = read_json(manifest_path)
    manifest["steps"]["routing"]["status"] = "failed"
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="Incomplete"):
        validate_evidence(config.paths.output)


def test_corpus_builders_with_stubbed_sources(experiment, monkeypatch):
    from moe_pilot.workflows import build_corpus, build_heldout, build_subtopics

    experiment = replace(
        experiment,
        datasets=tuple(
            DatasetRevision(name, "a" * 40)
            for name in (
                "openai/openai_humaneval",
                "EleutherAI/hendrycks_math",
                "allenai/sciq",
                "abisee/cnn_dailymail",
                "papluca/language-identification",
                "claudios/code_search_net",
                "b-mc2/sql-create-context",
                "cais/mmlu",
            )
        ),
    )

    class Rows(list):
        def select(self, indices):
            return Rows(self[i] for i in indices)

    def load_dataset(name, *args, **kwargs):
        assert kwargs["revision"] == "a" * 40
        labels = ["de", "fr", "es", "zh", "ru", "ar", "hi"] if "language" in name else ["en"]
        return Rows(
            {
                key: f"{name} {args} {label} {index} " + "source text " * 20
                for key in (
                    "prompt",
                    "canonical_solution",
                    "problem",
                    "solution",
                    "support",
                    "article",
                    "text",
                    "whole_func_string",
                    "context",
                    "answer",
                    "question",
                )
            }
            | {"labels": label, "choices": ["one", "two"]}
            for label in labels
            for index in range(180)
        )

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(
            load_dataset=load_dataset,
            concatenate_datasets=lambda groups: Rows(row for group in groups for row in group),
        ),
    )
    build_corpus.run(experiment)
    build_heldout.run(experiment)
    build_subtopics.run(experiment)
    data = experiment.paths.data
    corpus, heldout = read_json(data / "corpus.json"), read_json(data / "heldout.json")
    assert len(corpus) == 500
    assert len(heldout) == 200
    assert not {r["text"] for r in corpus} & {r["text"] for r in heldout}
    assert len(read_json(data / "subtopic_corpus.json")) == 19 * 60
    sources = json.loads((data / "subtopic_sources.json").read_text())
    assert all(source["n"] == 60 for source in sources["groups"].values())
    assert all(source["revision"] == "a" * 40 for source in sources["datasets"])
