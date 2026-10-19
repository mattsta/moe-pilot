"""Ablate ondomain ctrl pilot workflow."""


def main(model_dir="models/olmoe", control_candidates=6000):
    # Second control family: random NON-specialist experts matched on ON-DOMAIN routing mass (analysis corpus),
    # excluding the target domain's top-8 specialists at that layer. Checkpointed via run_ckpt (same base & caches).
    import os

    import numpy as np
    import pandas as pd

    from moe_pilot.artifacts import read_json, write_json
    from moe_pilot.controls import matched_experts, specialist_ranking
    from moe_pilot.engine import Engine

    M = Engine(model_dir)

    OUT = "results/ablations"
    doms = ["code", "math", "multilingual", "prose", "science"]
    R = np.load("routing.npz")
    topi = R["topi"].astype(int)
    topw = R["topw"].astype(np.float32)
    rdom = R["dom"]
    rdoms = list(R["doms"])
    mass = np.zeros((16, 5, 64))
    for l in range(16):
        for i, d in enumerate(doms):
            m = rdom == rdoms.index(d)
            np.add.at(mass[l, i], topi[l][m].ravel(), topw[l][m].ravel())
            mass[l, i] /= m.sum()
    de = pd.read_csv("results/expert_domain_distribution.csv")

    def spec_rank(l, d):
        return specialist_ranking(de, l, d)

    base = read_json(f"{OUT}/conditions.json")
    meta = pd.read_csv(f"{OUT}/conditions_expanded.csv")
    conds = {}
    rows = []

    def matched(l, S, d, seed):
        di = doms.index(d)
        return matched_experts(
            mass[l, di],
            S,
            seed,
            random_seed=5000 + 1000 * seed + l * 7 + len(S),
            exclude=spec_rank(l, d)[:8],
            candidates=control_candidates,
        )

    spec = meta[(~meta.control) & (meta.name == meta.canonical)]
    for _, r in spec.iterrows():
        if r.kind == "joint" and (r.k == 1 or (r.scope == "all" and r.k == 8)):
            continue
        m = {int(k): v for k, v in base[r["name"]].items()}
        for s in range(3):
            mm = {l: matched(l, v, r.domain, s) for l, v in m.items()}
            n = f"{r['name']}_odctrl{s}"
            conds[n] = mm
            di = doms.index(r.domain)
            rows.append(
                dict(
                    name=n,
                    parent=r["name"],
                    domain=r.domain,
                    seed=s,
                    target_on_mass=sum(mass[l, di, v].sum() for l, v in m.items()),
                    ctrl_on_mass=sum(mass[l, di, v].sum() for l, v in mm.items()),
                )
            )
    pd.DataFrame(rows).to_csv(f"{OUT}/conditions_ondomain_ctrl.csv", index=False)
    write_json(f"{OUT}/conditions_ondomain_ctrl.json", conds)
    print(
        len(conds),
        "conds",
        sum(16 - min(c) for c in conds.values()),
        "cond-layers",
        flush=True,
    )
    H = read_json("heldout.json")
    B = M.batch([M.tok(x["text"]).input_ids for x in H], 64)
    need = sorted(set(min(c) for c in conds.values()))
    missing = [s for s in need if not os.path.exists(f"{OUT}/cache/h_in_L{s}.npy")]
    print("missing caches", missing, flush=True)
    M.run_ckpt(B, conds, OUT, group=10)
    print("ALL DONE", flush=True)
