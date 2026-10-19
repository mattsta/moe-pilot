"""Act extra pilot workflow."""

import torch


@torch.inference_mode()
def main(model_dir="models/olmoe"):
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F

    from moe_pilot.model import ModelStore

    store = ModelStore(model_dir)

    XP = torch.from_numpy(np.load("probe_moe_inputs.npy"))
    E = store.config.num_experts
    iu = np.triu_indices(E, 1)
    rows = []
    for l in range(len(XP)):
        G = store.expert_weights(l, "gate")
        U = store.expert_weights(l, "up")
        D = store.expert_weights(l, "down")
        Y = torch.stack(
            [(F.silu(XP[l] @ G[e].T) * (XP[l] @ U[e].T)) @ D[e].T for e in range(E)]
        )  # E,P,H

        def cosm(Y):
            Yn = F.normalize(Y, dim=-1)
            return torch.einsum("eph,fph->efp", Yn, Yn).mean(-1).numpy()[iu]

        Yc = Y - Y.mean(0, keepdim=True)
        m = F.normalize(Y.mean(0), dim=-1)  # per-token mean-expert direction
        share_mean = ((Y * m[None]).sum(-1) ** 2 / (Y.pow(2).sum(-1))).mean().item()
        en = Y.pow(2).sum((0, 1))
        top1 = (en.max() / en.sum()).item()
        topdim = int(en.argmax())
        c = cosm(Y)
        cc = cosm(Yc)
        rows.append(
            dict(
                layer=l,
                mean_cos_raw=c.mean(),
                mean_cos_centered=cc.mean(),
                mean_abs_cos_centered=np.abs(cc).mean(),
                max_cos_centered=cc.max(),
                frac_energy_along_mean_expert_dir=share_mean,
                top1_hidden_dim_energy_share=top1,
                top1_hidden_dim=topdim,
            )
        )
        print(rows[-1], flush=True)
        del G, U, D
    pd.DataFrame(rows).to_csv("results/activation_orthogonality_extra.csv", index=False)
