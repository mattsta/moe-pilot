"""Specialist ranking and routing-mass-matched ablation controls."""

import numpy as np


def specialist_ranking(distribution, layer, domain):
    selected = distribution[distribution.layer == layer]
    ranked = selected.sort_values(f"P({domain}|e)", ascending=False)
    eligible = ranked[ranked[f"frac_{domain}_tokens_routed"] >= 0.05].expert.tolist()
    return eligible + [expert for expert in ranked.expert if expert not in eligible]


def matched_experts(mass, selected, seed, *, random_seed, exclude=(), candidates=4000):
    """Match total mass while excluding selected and forbidden experts.

    Singleton controls choose the seed-th closest candidate. Larger masks
    sample subsets and retain the closest total mass. Sampling and deterministic
    tie ordering preserve the original pilot procedure.
    """
    if candidates < 1:
        raise ValueError("control candidates must be positive")
    if not selected:
        raise ValueError("At least one selected expert is required")
    forbidden = set(selected) | set(exclude)
    pool = [expert for expert in range(len(mass)) if expert not in forbidden]
    if len(pool) < len(selected):
        raise ValueError("Insufficient eligible control experts")
    target = mass[list(selected)].sum()
    if len(selected) == 1:
        ordered = sorted(pool, key=lambda expert: abs(mass[expert] - target))
        if not 0 <= seed < len(ordered):
            raise ValueError("control_seeds exceeds the available singleton control pool")
        return [ordered[seed]]
    random = np.random.default_rng(random_seed)
    subsets = [random.choice(pool, len(selected), replace=False) for _ in range(candidates)]
    best = min(subsets, key=lambda subset: abs(mass[subset].sum() - target))
    return sorted(int(expert) for expert in best)
