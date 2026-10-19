"""Run every declared intervention through the same checkpointed engine."""

import numpy as np

from moe_pilot.artifacts import load_corpus
from moe_pilot.checkpoints import atomic_array
from moe_pilot.engine import Engine
from moe_pilot.interventions import Conditions


def run(config, *, device=None):
    paths = config.paths
    definition = Conditions.load(paths.ablations / "conditions.json")
    corpus = load_corpus(paths.data / "corpus.json")
    heldout = load_corpus(paths.data / "heldout.json")
    if {r["text"].strip() for r in corpus} & {r["text"].strip() for r in heldout}:
        raise ValueError("Held-out texts overlap the analysis corpus")
    if set(definition.domains) != {r["domain"] for r in heldout}:
        raise ValueError("Held-out domains must match the analysis domains")
    engine = Engine(paths.model, device=device or config.execution.device)
    if (definition.layers, definition.experts, definition.top_k) != (engine.L, engine.E, engine.K):
        raise ValueError("Condition dimensions differ from the configured model")
    batch = engine.batch(
        [engine.tok(r["text"]).input_ids for r in heldout], config.execution.max_tokens
    )
    checkpoint = paths.ablations / "checkpoint"
    engine.run_ckpt(
        batch,
        {item.name: item.mask for item in definition.items},
        checkpoint,
        group=config.execution.condition_group,
    )
    domain_path = paths.ablations / "token_domain.npy"
    doc_path = paths.ablations / "token_doc.npy"
    atomic_array(
        domain_path, np.array([definition.domains.index(heldout[d]["domain"]) for d in batch.doc])
    )
    atomic_array(doc_path, batch.doc)
    return [
        domain_path,
        doc_path,
        checkpoint / "manifest.json",
        checkpoint / "valid.npy",
        checkpoint / "losses/base.npy",
        *[checkpoint / f"losses/{item.name}.npy" for item in definition.items],
    ]
