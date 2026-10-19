"""Build specialists and both control families together from routing evidence."""

import numpy as np
import pandas as pd

from moe_pilot.controls import matched_experts, specialist_ranking
from moe_pilot.interventions import Condition, Conditions


def run(config):
    paths = config.paths
    paths.prepare_outputs()
    with np.load(paths.routing) as routing:
        indices = routing["topi"].astype(int)
        weights = routing["topw"].astype(np.float32)
        domains = tuple(str(value) for value in routing["doms"])
        labels = routing["dom"]
        layers, tokens, top_k = indices.shape
        experts = routing["probs"].shape[-1]
    counts = config.ablations.expert_counts
    if any(count > min(experts - top_k, experts // 2) for count in counts):
        raise ValueError(
            "expert_counts must leave top-k active experts and equally sized disjoint controls"
        )
    overall = np.zeros((layers, experts))
    on_domain = np.zeros((layers, len(domains), experts))
    for layer in range(layers):
        np.add.at(overall[layer], indices[layer].ravel(), weights[layer].ravel())
        overall[layer] /= tokens
        for domain in range(len(domains)):
            selected = labels == domain
            np.add.at(
                on_domain[layer, domain],
                indices[layer][selected].ravel(),
                weights[layer][selected].ravel(),
            )
            on_domain[layer, domain] /= selected.sum()
    distribution = pd.read_csv(paths.tables / "expert_domain_distribution.csv")

    def describe(name, domain, scope, count, family, mask, parent=None, seed=None):
        domain_index = domains.index(domain)
        return Condition(
            name,
            domain,
            scope,
            count,
            family,
            mask,
            global_mass=float(
                sum(overall[layer, experts].sum() for layer, experts in mask.items())
            ),
            on_domain_mass=float(
                sum(
                    on_domain[layer, domain_index, experts].sum() for layer, experts in mask.items()
                )
            ),
            parent=parent,
            seed=seed,
        )

    selected_layers = config.analysis.selected_layers(layers)
    scopes = [(f"layer-{layer}", (layer,)) for layer in selected_layers]
    if layers > 1:
        scopes.append(("all-layers", tuple(range(layers))))
    items = []
    for domain_index, domain in enumerate(domains):
        rankings = {
            layer: specialist_ranking(distribution, layer, domain) for layer in range(layers)
        }
        for count in counts:
            for scope, affected_layers in scopes:
                mask = {
                    layer: [int(e) for e in rankings[layer][:count]] for layer in affected_layers
                }
                name = f"domain{domain_index}_count{count}_{scope}"
                items.append(describe(name, domain, scope, count, "specialist", mask))
                for family, candidate_count in (
                    ("global-control", config.ablations.control_candidates),
                    ("on-domain-control", config.ablations.on_domain_candidates),
                ):
                    for seed in range(config.ablations.control_seeds):
                        control = {}
                        for layer in affected_layers:
                            mass = (
                                overall[layer]
                                if family == "global-control"
                                else on_domain[layer, domain_index]
                            )
                            excluded = (
                                ()
                                if family == "global-control"
                                else rankings[layer][: max(top_k, count)]
                            )
                            control[layer] = matched_experts(
                                mass,
                                mask[layer],
                                seed,
                                exclude=excluded,
                                candidates=candidate_count,
                                random_seed=config.analysis.seed
                                + domain_index * 100000
                                + layer * 1000
                                + count * 10
                                + seed,
                            )
                        items.append(
                            describe(
                                f"{name}_{family}{seed}",
                                domain,
                                scope,
                                count,
                                family,
                                control,
                                name,
                                seed,
                            )
                        )
    filename = paths.ablations / "conditions.json"
    Conditions(domains, layers, experts, top_k, tuple(items)).save(filename)
    return [filename]
