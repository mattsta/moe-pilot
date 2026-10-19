import pytest

from moe_pilot.study_formats import paired_chat_rows


class CharacterTokenizer:
    chat_template = "native-test-wrapper"

    def apply_chat_template(self, messages, **kwargs):
        return "USER\n" + messages[0]["content"] + "\nASSISTANT\n"

    def __call__(self, text, **kwargs):
        return {
            "input_ids": list(map(ord, text)),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def test_native_wrapper_preserves_content_labels_and_identical_token_span():
    prefix = "Question: Which value?\nA. one\nB. two\nAnswer:"
    row = {
        "id": "question-1",
        "group_id": "group-1",
        "domain": "science",
        "source": "source",
        "text": prefix + " B",
        "answer_start": len(prefix),
        "answer_index": 1,
    }
    tokenizer = CharacterTokenizer()
    plain, chat = paired_chat_rows(tokenizer, [row])
    assert plain["source_id"] == chat["source_id"] == row["id"]
    assert plain["sample_id"] == chat["sample_id"]
    assert plain["group_id"] == chat["group_id"]
    assert plain["text"][plain["answer_start"] :] == chat["text"][chat["answer_start"] :] == " B"
    assert chat["add_special_tokens"] is False
    assert plain["alignment_coverage"] == chat["alignment_coverage"] == 1
    sequences = [
        [tokenizer(item["text"])["input_ids"][index] for index in item["routing_positions"]]
        for item in (plain, chat)
    ]
    assert sequences[0] == sequences[1] == list(map(ord, prefix))


def test_missing_native_template_cannot_silently_use_a_made_up_prompt():
    tokenizer = CharacterTokenizer()
    tokenizer.chat_template = None
    with pytest.raises(ValueError, match="recorded chat template"):
        paired_chat_rows(tokenizer, [])
