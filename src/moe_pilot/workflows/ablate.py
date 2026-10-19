"""Ablate pilot workflow."""


def main(model_dir="models/olmoe", control_candidates=4000):
    from pathlib import Path

    Path("results/ablations").mkdir(parents=True, exist_ok=True)
    import json

    import numpy as np
    import pandas as pd

    from moe_pilot.artifacts import write_json
    from moe_pilot.controls import matched_experts, specialist_ranking

    R = np.load("routing.npz")
    topi = R["topi"].astype(int)
    topw = R["topw"].astype(np.float32)
    N = topi.shape[1]
    mass = np.zeros((16, 64))
    for l in range(16):
        np.add.at(mass[l], topi[l].ravel(), topw[l].ravel())
    mass /= N  # mean router weight per token routed to each expert (analysis corpus); sums to top-8 mass
    de = pd.read_csv("results/expert_domain_distribution.csv")
    doms = ["code", "math", "multilingual", "prose", "science"]

    def spec_rank(l, d):
        return specialist_ranking(de, l, d)

    def matched(l, S, seed):
        return matched_experts(
            mass[l],
            S,
            seed,
            random_seed=1000 * seed + l * 7 + len(S),
            candidates=control_candidates,
        )

    conds = {}
    meta = []

    def add(name, mask, **kw):
        key = json.dumps({str(k): sorted(v) for k, v in sorted(mask.items())})
        for m in meta:
            if m["key"] == key:
                m.setdefault("aliases", []).append(name)
                return m["name"]
        conds[name] = {l: list(map(int, v)) for l, v in mask.items()}
        meta.append(
            dict(
                name=name,
                key=key,
                on_mass=sum(mass[l, v].sum() for l, v in mask.items()),
                **kw,
            )
        )
        return name

    singles = [
        (7, 45, "code"),
        (7, 35, "math"),
        (7, 27, "multilingual"),
        (7, 47, "prose"),
        (7, 63, "science"),
        (15, 8, "code"),
        (15, 30, "code"),
        (15, 3, "math"),
        (15, 12, "multilingual"),
        (15, 0, "multilingual"),
        (15, 25, "prose"),
        (15, 59, "prose"),
        (15, 38, "science"),
        (0, 21, "code"),
        (0, 9, "science"),
    ]
    for l, e, d in singles:
        add(
            f"single_L{l}_E{e}",
            {l: [e]},
            kind="single",
            domain=d,
            scope=f"L{l}",
            k=1,
            control=False,
            seed=-1,
        )
        for s in range(3):
            add(
                f"single_L{l}_E{e}_rand{s}",
                {l: matched(l, [e], s)},
                kind="single",
                domain=d,
                scope=f"L{l}",
                k=1,
                control=True,
                seed=s,
            )
    for d in doms:
        for k in (1, 4, 8):
            for scope in ("L7", "L15", "all"):
                layers = [7] if scope == "L7" else [15] if scope == "L15" else range(16)
                mask = {l: spec_rank(l, d)[:k] for l in layers}
                add(
                    f"joint_{d}_k{k}_{scope}",
                    mask,
                    kind="joint",
                    domain=d,
                    scope=scope,
                    k=k,
                    control=False,
                    seed=-1,
                )
                for s in range(3):
                    add(
                        f"joint_{d}_k{k}_{scope}_rand{s}",
                        {l: matched(l, mask[l], s) for l in layers},
                        kind="joint",
                        domain=d,
                        scope=scope,
                        k=k,
                        control=True,
                        seed=s,
                    )
    pd.DataFrame(meta).to_csv("results/ablations/conditions.csv", index=False)
    write_json("results/ablations/conditions.json", conds)
    print(
        len(conds),
        "conditions",
        sum(16 - min(c) for c in conds.values()),
        "condition-layers",
        flush=True,
    )
