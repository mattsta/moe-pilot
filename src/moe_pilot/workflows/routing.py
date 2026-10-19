"""Produce routing and geometry artifacts through the shared inference engine."""

import numpy as np
import pandas as pd

from moe_pilot.artifacts import load_corpus
from moe_pilot.config import Experiment
from moe_pilot.engine import Engine
from moe_pilot.probes import OrthogonalityProbe


def run(config: Experiment):
    paths = config.paths
    paths.prepare_outputs()
    corpus = load_corpus(paths.data / "corpus.json")
    engine = Engine(paths.model)
    batch = engine.batch(
        [engine.tok(c["text"]).input_ids for c in corpus], config.execution.max_tokens
    )
    domains = sorted({c["domain"] for c in corpus})
    if len(domains) < 2:
        raise ValueError("Domain analysis requires at least two domains")
    domain = np.array([domains.index(corpus[d]["domain"]) for d in batch.doc])
    random = np.random.default_rng(config.analysis.seed)
    probe_indices = np.concatenate(
        [
            random.choice(
                indices, min(config.analysis.probe_tokens_per_domain, len(indices)), replace=False
            )
            for d in range(len(domains))
            if len(indices := np.where(domain == d)[0])
        ]
    )
    probe = OrthogonalityProbe(probe_indices, config.analysis.seed)
    losses, valid, routing = engine.run(batch, {}, log_routing=True, on_layer=probe)
    np.savez_compressed(
        paths.routing,
        topi=np.stack([r[0] for r in routing]),
        topw=np.stack([r[1] for r in routing]),
        probs=np.stack([r[2] for r in routing]),
        tok_ids=batch.tok.numpy(),
        doc=batch.doc,
        pos=batch.pos,
        dom=domain,
        doms=np.array(domains),
        probe=probe_indices,
        loss=losses["base"],
        valid=valid,
    )
    np.savez_compressed(paths.orthogonality, **probe.matrices)
    inputs_path = paths.artifacts / "probe_inputs.npy"
    np.save(inputs_path, np.stack(probe.inputs))
    extra_path = paths.tables / "activation_orthogonality.csv"
    pd.DataFrame(probe.extra).to_csv(extra_path, index=False)
    return [paths.routing, paths.orthogonality, inputs_path, extra_path]
