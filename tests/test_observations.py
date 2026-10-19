import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from moe_pilot.engine import Engine
from moe_pilot.observations import observe_documents
from moe_pilot.smoke import tiny_model
from moe_pilot.study_data import render_question


@pytest.fixture
def observation_inputs(tmp_path):
    directory = tmp_path / "model"
    tiny_model(directory)
    words = [
        "[UNK]",
        "A",
        "B",
        "Question",
        "Answer",
        "Which",
        "value",
        "alpha",
        "beta",
        "one",
        "two",
    ]
    tokenizer = Tokenizer(
        WordLevel(dict(zip(words, range(len(words)), strict=True)), unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(
        directory
    )
    rows = []
    for index in range(4):
        question = "Which value " + ("alpha " if index % 2 else "beta ") * (index + 1)
        choices, answer = ["one", "two"], index % 2
        text, start = render_question(question, choices, answer)
        rows.append(
            {
                "id": str(index),
                "group_id": str(index),
                "source": "example/source",
                "origin": "fixture",
                "domain": "first" if index < 2 else "second",
                "text": text,
                "question": question,
                "choices": choices,
                "answer_index": answer,
                "answer_start": start,
            }
        )
    return Engine(directory, device="cpu", weight_cache="resident"), rows


def test_microbatching_preserves_document_routing_losses_and_norm_samples(
    observation_inputs, tmp_path
):
    engine, rows = observation_inputs
    previous = torch.get_rng_state().clone()
    complete, _ = observe_documents(engine, rows, max_tokens=128, layers=(0, 1), seed=17)
    chunks, metadata = observe_documents(
        engine,
        rows,
        max_tokens=128,
        layers=(0, 1),
        seed=17,
        microbatch_documents=1,
        output=tmp_path / "observations.npz",
    )
    for name in ("mass", "usage", "norm", "loss", "answer_loss", "features"):
        np.testing.assert_allclose(chunks[name], complete[name], atol=2e-5, rtol=2e-5)
    np.testing.assert_array_equal(torch.get_rng_state(), previous)
    assert chunks["mass"].shape == (2, 4, 4)
    np.testing.assert_allclose(chunks["usage"].sum(-1), engine.K)
    assert np.isfinite(chunks["answer_loss"]).all()
    assert metadata["resident_weight_bytes"] > 0
    assert (tmp_path / "observations.json").is_file()


def test_observations_reject_truncating_answers_and_cross_model_frequencies(observation_inputs):
    engine, rows = observation_inputs
    with pytest.raises(ValueError, match="context limit"):
        observe_documents(engine, rows, max_tokens=3, layers=(0,), seed=17)
    with pytest.raises(ValueError, match="vocabulary"):
        observe_documents(
            engine, rows, max_tokens=128, layers=(0,), seed=17, discovery_counts=np.array([1, 2, 3])
        )


def test_choice_prediction_cannot_read_the_given_answer(observation_inputs):
    engine, rows = observation_inputs
    paired = []
    for answer in (0, 1):
        text, start = render_question(rows[0]["question"], rows[0]["choices"], answer)
        paired.append(
            {
                **rows[0],
                "id": str(answer),
                "text": text,
                "answer_start": start,
                "answer_index": answer,
            }
        )
    arrays, _ = observe_documents(engine, paired, max_tokens=128, layers=(0,), seed=17)
    assert arrays["choice_accuracy"].sum() == 1
    np.testing.assert_allclose(np.exp(arrays["choice_log_probability"]).sum(), 1, atol=1e-6)
