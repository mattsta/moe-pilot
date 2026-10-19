"""Offline integration of research artifact production and corpus orchestration."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from moe_pilot.artifacts import read_json, write_json
from moe_pilot.cli import main
from moe_pilot.smoke import tiny_model


@pytest.fixture
def research_workspace(tmp_path):
    model_dir = tmp_path / "models/olmoe"
    tiny_model(model_dir, layers=16, experts=64)
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
        for offset in range(3)
    ]
    write_json(tmp_path / "corpus.json", corpus)
    write_json(tmp_path / "heldout.json", corpus)
    write_json(
        tmp_path / "subtopic_corpus.json",
        [
            {**document, "subtopic": f"topic{index % 2}"}
            for index, document in enumerate(corpus)
            if document["domain"] != "prose"
        ],
    )
    return tmp_path


def test_forward_analysis_ablations_and_subtopic_reports(research_workspace):
    root = research_workspace
    arguments = ["--workspace", str(root), "--threads", "2"]
    for command in ("forward", "analyze", "activation-extra"):
        assert main([*arguments, command]) == 0
    routing = np.load(root / "routing.npz")
    assert routing["topi"].shape == (16, 720, 2)
    assert pd.read_csv(root / "results/layer_domain_metrics.csv").shape[0] == 80
    assert pd.read_csv(root / "results/activation_orthogonality_extra.csv").shape[0] == 16
    assert (root / "plots/jsd_by_layer.png").stat().st_size > 1000

    assert main([*arguments, "build-conditions", "--control-candidates", "16"]) == 0
    generated = read_json(root / "results/ablations/conditions.json")
    assert len(generated) > 20
    for mask in generated.values():
        assert all(0 <= int(layer) < 16 for layer in mask)
        assert all(0 <= expert < 64 for experts in mask.values() for expert in experts)

    # A small valid condition family exercises the complete held-out reporting
    # path without running the original hundreds-of-conditions experiment.
    directory = root / "results/ablations"
    names = ["single_L0_E21", *[f"single_L0_E21_rand{i}" for i in range(3)]]
    write_json(directory / "conditions.json", {n: {"0": [21 + i]} for i, n in enumerate(names)})
    pd.DataFrame(
        [{"name": name, "aliases": "[]", "control": index > 0} for index, name in enumerate(names)]
    ).to_csv(directory / "conditions.csv", index=False)
    for command in ("ablate", "analyze-ablations", "ondomain-controls", "analyze-ablations"):
        assert main([*arguments, command]) == 0
    first = pd.read_csv(directory / "delta_loss_long.csv")
    assert len(first) == 20
    assert main([*arguments, "ablate"]) == 0
    pd.testing.assert_frame_equal(first, pd.read_csv(directory / "delta_loss_long.csv"))
    assert len(pd.read_csv(directory / "ablation_summary.csv")) == 1
    assert len(pd.read_csv(directory / "ablation_vs_ondomain_mass_controls.csv")) == 1
    assert main([*arguments, "subtopics"]) == 0
    assert len(pd.read_csv(root / "results/subtopics/layer_within_vs_between_jsd.csv")) == 64
    assert (root / "plots/subtopics/jsd_within_vs_between.png").stat().st_size > 1000


def test_corpus_builders_with_stubbed_sources(tmp_path, monkeypatch):
    from moe_pilot.workflows import build_corpus, build_heldout, build_subtopics

    class Rows(list):
        def select(self, indices):
            return Rows(self[i] for i in indices)

    def load_dataset(name, *args, **kwargs):
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
    monkeypatch.chdir(tmp_path)
    build_corpus.main()
    build_heldout.main()
    build_subtopics.main()
    corpus, heldout = read_json("corpus.json"), read_json("heldout.json")
    assert len(corpus) == 500
    assert len(heldout) == 200
    assert not {r["text"] for r in corpus} & {r["text"] for r in heldout}
    assert len(read_json("subtopic_corpus.json")) == 19 * 60
    sources = json.loads((tmp_path / "results/subtopics/corpus_sources.json").read_text())
    assert all(source["n"] == 60 for source in sources.values())
