"""Ablate run pilot workflow."""


def main(model_dir="models/olmoe"):
    # Checkpointed runner: reuses results/ablations/conditions.json (built by ablate.py). Appends per-condition
    # per-domain delta-loss rows to results/ablations/delta_loss_long.csv as each group finishes.
    import os

    import numpy as np
    import pandas as pd

    from moe_pilot.artifacts import read_json
    from moe_pilot.engine import Engine

    M = Engine(model_dir)

    OUT = "results/ablations"
    doms = ["code", "math", "multilingual", "prose", "science"]
    conds = {
        n: {int(k): v for k, v in c.items()} for n, c in read_json(f"{OUT}/conditions.json").items()
    }
    H = read_json("heldout.json")
    B = M.batch([M.tok(x["text"]).input_ids for x in H], 64)
    dom = np.array([doms.index(H[d]["domain"]) for d in B.doc])
    CSV = f"{OUT}/delta_loss_long.csv"

    def on_done(names):
        valid = np.load(f"{OUT}/valid.npy")
        base = np.load(f"{OUT}/losses/base.npy")
        rows = []
        for n in names:
            if n == "base":
                continue
            ls = np.load(f"{OUT}/losses/{n}.npy")
            d = ls - base
            for i, dn in enumerate(doms):
                m = valid & (dom == i)
                rows.append(
                    dict(
                        condition=n,
                        domain_eval=dn,
                        base_loss=base[m].mean(),
                        abl_loss=ls[m].mean(),
                        delta=d[m].mean(),
                        n_tokens=int(m.sum()),
                    )
                )
        if rows:
            fresh = pd.DataFrame(rows)
            if os.path.exists(CSV):
                fresh = pd.concat([pd.read_csv(CSV), fresh], ignore_index=True)
            fresh.drop_duplicates(["condition", "domain_eval"], keep="last").to_csv(
                CSV, index=False
            )

    M.run_ckpt(B, conds, OUT, group=10, on_done=on_done)
    np.save(f"{OUT}/token_domain.npy", dom)
    np.save(f"{OUT}/token_doc.npy", B.doc)
    # Repair any summary rows missed by a crash between saving losses and the callback.
    on_done(list(conds))
    print("ALL DONE", flush=True)
