"""Document-level routing, expert function norms, and loss observations."""

import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from moe_pilot.artifacts import write_json
from moe_pilot.devices import to_numpy
from moe_pilot.statistics import document_means

FEATURES = (
    "log_tokens",
    "log_bytes",
    "digit_fraction",
    "punctuation_fraction",
    "non_ascii_fraction",
    "choice_count",
    "mean_log_discovery_token_frequency",
)


class ExpertNormProbe:
    """Raw expert-output norms on one deterministic valid input per document."""

    def __init__(self, layers, positions):
        self.layers = set(layers)
        self.positions = positions
        self.values = {}

    def __call__(self, layer, inputs, weights):
        if layer not in self.layers:
            return
        sample = inputs[torch.tensor(self.positions, device=inputs.device)]
        norms = []
        for gate, up, down in zip(weights["G"], weights["U"], weights["D"], strict=True):
            output = (F.silu(sample @ gate.T) * (sample @ up.T)) @ down.T
            norms.append(to_numpy(output.norm(dim=-1)))
        self.values[layer] = np.stack(norms, axis=1)


def encode_rows(engine, rows, max_tokens):
    ids, target_masks = [], []
    for row in rows:
        encoded = engine.tok(row["text"], return_offsets_mapping=True)
        tokens = encoded["input_ids"]
        if len(tokens) < 2 or len(tokens) > max_tokens:
            raise ValueError(
                f"Study observation {row['id']} does not fit its declared context limit"
            )
        offsets = encoded["offset_mapping"]
        answer = [False] * len(tokens)
        for index in range(1, len(tokens)):
            if offsets[index][1] > row["answer_start"]:
                answer[index - 1] = True
        if not any(answer):
            raise ValueError("A study observation has no valid answer target")
        ids.append(tokens)
        target_masks.append(np.array(answer, dtype=bool))
    return ids, target_masks


@torch.inference_mode()
def observe_documents(
    engine,
    rows,
    *,
    max_tokens,
    layers,
    seed,
    microbatch_documents=4,
    discovery_counts=None,
    output=None,
):
    """Collect paired document observations without truncating target content."""
    ids, answer_masks = encode_rows(engine, rows, max_tokens)
    layers = tuple(layers)
    if (
        not rows
        or not layers
        or len(layers) != len(set(layers))
        or any(not 0 <= layer < engine.L for layer in layers)
    ):
        raise ValueError("Observations require nonempty records and valid layers")
    if type(microbatch_documents) is not int or microbatch_documents < 0:
        raise ValueError("microbatch_documents must be a nonnegative integer")
    count, experts = len(rows), engine.E
    mass = np.zeros((len(layers), count, experts))
    usage, norms = np.zeros_like(mass), np.zeros_like(mass)
    loss, answer_loss = np.zeros(count), np.zeros(count)
    token_counts = np.array([len(tokens) for tokens in ids])
    unigram = np.bincount(np.concatenate(ids), minlength=engine.cfg.vocab_size)
    frequency = unigram if discovery_counts is None else discovery_counts
    if (
        frequency.shape != (engine.cfg.vocab_size,)
        or not np.isfinite(frequency).all()
        or (frequency < 0).any()
    ):
        raise ValueError("Discovery token frequencies must match the model vocabulary")
    features = []
    for row, tokens in zip(rows, ids, strict=True):
        text = row["text"]
        features.append(
            [
                np.log(len(tokens)),
                np.log(len(text.encode())),
                sum(character.isdigit() for character in text) / len(text),
                sum(unicodedata.category(character).startswith("P") for character in text)
                / len(text),
                sum(ord(character) > 127 for character in text) / len(text),
                len(row["choices"]),
                float(np.log1p(frequency[tokens]).mean()),
            ]
        )
    width = microbatch_documents or count
    for start in range(0, count, width):
        stop = min(start + width, count)
        batch = engine.batch(ids[start:stop], max_tokens)
        positions = []
        offset = 0
        for row, tokens in zip(rows[start:stop], ids[start:stop], strict=True):
            number = int.from_bytes(
                hashlib.sha256(f"{seed}:{row['id']}".encode()).digest()[:8], "big"
            )
            positions.append(offset + number % (len(tokens) - 1))
            offset += len(tokens)
        probe = ExpertNormProbe(layers, positions)
        results, valid, routes = engine.run(
            batch, {}, log_routing=True, on_layer=probe, verbose=False
        )
        loss[start:stop] = document_means(results["base"], batch.doc, valid)
        answer_targets = np.concatenate(answer_masks[start:stop]) & valid
        answer_loss[start:stop] = document_means(results["base"], batch.doc, answer_targets)
        for position, layer in enumerate(layers):
            indices, gates, _ = routes[layer]
            selected_docs = np.repeat(batch.doc[valid], engine.K)
            local_mass = np.zeros((stop - start, experts))
            local_usage = np.zeros_like(local_mass)
            np.add.at(local_mass, (selected_docs, indices[valid].ravel()), gates[valid].ravel())
            np.add.at(local_usage, (selected_docs, indices[valid].ravel()), 1)
            denominators = (token_counts[start:stop] - 1)[:, None]
            mass[position, start:stop] = local_mass / denominators
            usage[position, start:stop] = local_usage / denominators
            norms[position, start:stop] = probe.values[layer]
        print(f"Observed documents {stop}/{count} on {engine.selection.backend}", flush=True)
    arrays = {
        "layers": np.array(layers),
        "mass": mass,
        "usage": usage,
        "norm": norms,
        "loss": loss,
        "answer_loss": answer_loss,
        "token_counts": token_counts,
        "unigram": unigram,
        "features": np.array(features),
        "feature_names": np.array(FEATURES),
    }
    metadata = {
        "schema": 1,
        "model_fingerprint": engine.store.fingerprint(),
        "device": engine.selection.record(),
        "weight_cache": engine.weight_cache,
        "resident_weight_bytes": engine.resident_weight_bytes,
        "token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "norm_sampling": "one uniform deterministic valid position per document; unweighted expert outputs",
        "routing_weighting": "equal document means; selected gate mass and selected frequency",
        "records": [
            {key: row[key] for key in ("id", "domain", "source", "origin", "group_id")}
            for row in rows
        ],
    }
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **arrays)
        write_json(output.with_suffix(".json"), metadata)
    return arrays, metadata
