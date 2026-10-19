"""Paired native-chat protocols with an exactly aligned question-token span."""

import hashlib
import json
from difflib import SequenceMatcher

FORMAT_PLAN = {
    "schema": 1,
    "status": "exploratory follow-up selected after plain-prompt stage evaluation",
    "factor": "plain versus checkpoint-native user/assistant wrapper; identical question and response-label strings",
    "alignment": "longest identical contiguous question-token span, at least 90% of the plain prefix",
    "sampling": "same deterministic token index inside the aligned span for each paired content group",
    "outcomes": "paired forced-choice accuracy, normalized choice log loss, aligned-span next-token loss, and selected-gate routing divergence",
    "uncertainty": "10000 paired content-group bootstrap replicates stratified by source and domain",
    "limits": "protocol effects within a fixed checkpoint and cohort; not a training-algorithm effect or official benchmark score",
}


def paired_chat_rows(tokenizer, rows):
    """Keep question/choices fixed; align common tokens before the given answer.

    Response-label strings are held fixed across protocols. Question routing
    and loss can be compared on the same contiguous token sequence despite
    boundary tokenization changes introduced by the wrapper.
    """
    if not tokenizer.chat_template:
        raise ValueError("Native-chat study requires the checkpoint's recorded chat template")
    paired = []
    for row in rows:
        prompt = row["text"][: row["answer_start"]]
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        if not prompt or prefix.count(prompt) != 1:
            raise ValueError("Cannot identify one preserved question in the chat wrapper")
        plain = {
            **row,
            "id": row["id"] + "--plain",
            "source_id": row["id"],
            "sample_id": row["id"],
            "format_variant": "plain",
        }
        chat = {
            **row,
            "id": row["id"] + "--chat",
            "source_id": row["id"],
            "sample_id": row["id"],
            "format_variant": "chat",
            "text": prefix + row["text"][row["answer_start"] :],
            "answer_start": len(prefix),
            "add_special_tokens": False,
            "choice_separator": " ",
        }
        encoded, positions = [], []
        for item, start, stop in (
            (plain, 0, len(prompt)),
            (chat, prefix.index(prompt), prefix.index(prompt) + len(prompt)),
        ):
            result = tokenizer(
                item["text"],
                return_offsets_mapping=True,
                add_special_tokens=item.get("add_special_tokens", True),
            )
            indices = [
                index
                for index, (first, last) in enumerate(result["offset_mapping"])
                if first >= start
                and last <= stop
                and last > first
                and index < len(result["input_ids"]) - 1
            ]
            encoded.append(result["input_ids"])
            positions.append(indices)
        sequences = [
            [tokens[index] for index in indices]
            for tokens, indices in zip(encoded, positions, strict=True)
        ]
        match = SequenceMatcher(a=sequences[0], b=sequences[1], autojunk=False).find_longest_match()
        coverage = match.size / max(1, len(sequences[0]))
        if match.size < 2 or coverage < 0.9:
            raise ValueError(
                "Chat wrapping did not preserve a sufficiently long question-token span"
            )
        for item, indices, start in ((plain, positions[0], match.a), (chat, positions[1], match.b)):
            item["routing_positions"] = indices[start : start + match.size]
            if item["routing_positions"] != list(
                range(item["routing_positions"][0], item["routing_positions"][0] + match.size)
            ):
                raise ValueError("Aligned question spans must be contiguous in both protocols")
            item["alignment_coverage"] = coverage
        aligned_digest = hashlib.sha256(
            json.dumps([encoded[0][index] for index in plain["routing_positions"]]).encode()
        ).hexdigest()
        plain["aligned_tokens_sha256"] = chat["aligned_tokens_sha256"] = aligned_digest
        if [encoded[0][index] for index in plain["routing_positions"]] != [
            encoded[1][index] for index in chat["routing_positions"]
        ]:
            raise ValueError("Question-token alignment failed")
        paired.extend((plain, chat))
    return paired
