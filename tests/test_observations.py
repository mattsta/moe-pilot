import numpy as np
import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from moe_pilot.engine import Engine
from moe_pilot.observations import observe_documents
from moe_pilot.resources import ResourcePolicy
from moe_pilot.smoke import tiny_model
from moe_pilot.study_data import render_question
from moe_pilot.study_formats import paired_chat_rows


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
    return Engine(
        directory, device="cpu", weight_cache="resident", resources=ResourcePolicy(2, 4)
    ), rows


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


def test_intervention_observations_record_gate_changes_and_exact_restoration(observation_inputs):
    from moe_pilot.perturbations import Intervention

    engine, rows = observation_inputs
    arrays, metadata = observe_documents(
        engine,
        rows[:2],
        max_tokens=128,
        layers=(0,),
        seed=17,
        conditions={
            "remove": Intervention({0: [0]}, "remove"),
            "restore": Intervention({0: [0]}, "remove", 0),
        },
    )
    np.testing.assert_array_equal(arrays["condition_loss"][1], arrays["loss"])
    traces = metadata["intervention_diagnostics"]
    assert len(traces) == 1
    assert traces[0]["condition"] == "remove" and traces[0]["layer"] == 0
    assert traces[0]["fraction_changed_expert_sets"] == 0
    assert traces[0]["post_selected_mass_mean"] <= traces[0]["original_selected_mass_mean"]


def test_matched_question_spans_keep_probe_and_routing_invariant_to_batching(observation_inputs):
    engine, rows = observation_inputs
    engine.tok.chat_template = (
        "{% for message in messages %}USER {{ message['content'] }} ASSISTANT{% endfor %}"
    )
    paired = paired_chat_rows(engine.tok, rows[:2])
    full, _ = observe_documents(engine, paired, max_tokens=128, layers=(0,), seed=17)
    split, _ = observe_documents(
        engine, paired, max_tokens=128, layers=(0,), seed=17, microbatch_documents=1
    )
    for key in (
        "mass",
        "usage",
        "norm",
        "aligned_loss",
        "choice_accuracy",
        "choice_log_probability",
    ):
        np.testing.assert_allclose(full[key], split[key], atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(full["usage"].sum(-1), engine.K)
    assert full["routing_token_counts"][0] == full["routing_token_counts"][1]
    assert full["routing_token_counts"][2] == full["routing_token_counts"][3]
    reference = AutoModelForCausalLM.from_pretrained(
        engine.store.directory, local_files_only=True
    ).eval()
    with torch.inference_mode():
        for index, row in enumerate(paired):
            ids = torch.tensor(
                engine.tok(
                    row["text"], add_special_tokens=row.get("add_special_tokens", True)
                ).input_ids
            )
            logits = reference(ids[None], use_cache=False).logits[0]
            positions = np.array(row["routing_positions"])
            expected = F.cross_entropy(logits[positions[:-1]], ids[positions[1:]]).item()
            np.testing.assert_allclose(full["aligned_loss"][index], expected, atol=2e-5, rtol=2e-5)
