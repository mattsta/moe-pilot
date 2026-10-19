"""Analyze pilot workflow."""


def main(model_dir="models/olmoe"):
    import itertools
    import keyword
    import os
    import re

    import matplotlib
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer

    from moe_pilot.metrics import entropy_bits as H2
    from moe_pilot.metrics import jensen_shannon as jsd
    from moe_pilot.metrics import ratio

    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    R = np.load("routing.npz")
    O = np.load("ortho_raw.npz")
    topi, topw, probs, tok_ids, doc, dom = (
        R["topi"].astype(int),
        R["topw"].astype(np.float32),
        R["probs"].astype(np.float32),
        R["tok_ids"],
        R["doc"],
        R["dom"],
    )
    doms = list(R["doms"])
    ND = len(doms)
    L, N, K = topi.shape
    E = probs.shape[2]
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    def counts(l, dlab):
        C = np.zeros((ND, E))
        for d in range(ND):
            C[d] = np.bincount(topi[l][dlab == d].ravel(), minlength=E)
        return C

    rng = np.random.default_rng(0)
    ndoc = doc.max() + 1
    docdom = np.array([dom[doc == i][0] for i in range(ndoc)])
    NPERM = 20
    perm_labels = [rng.permutation(docdom)[doc] for _ in range(NPERM)]
    rows = []
    pair_rows = []
    exp_rows = []
    Cs = {}
    for l in range(L):
        C = counts(l, dom)
        Cs[l] = C
        Pd = C / C.sum(1, keepdims=True)
        Pall = Pd.mean(0)
        Pbal = ratio(Pd, Pd.sum(0, keepdims=True))  # Undefined for unused experts
        # permutation null (doc-level shuffle of domain labels)
        null_j = []
        null_si = []
        null_pair = []
        for pl in perm_labels:
            Cp = counts(l, pl)
            Pp = Cp / Cp.sum(1, keepdims=True)
            Pm = Pp.mean(0)
            Pb = ratio(Pp, Pp.sum(0, keepdims=True))
            null_j.append([jsd(Pp[d], Pm) for d in range(ND)])
            null_si.append(np.nanmean(1 - H2(Pb.T) / np.log2(ND)))
            null_pair.append(
                np.mean([jsd(Pp[a], Pp[b]) for a, b in itertools.combinations(range(ND), 2)])
            )
        null_j = np.array(null_j)
        ent_tok = H2(probs[l])  # per token full-router entropy (bits)
        top8 = [set(np.argsort(-Pd[d])[:8]) for d in range(ND)]
        for d in range(ND):
            m = dom == d
            rows.append(
                dict(
                    layer=l,
                    domain=doms[d],
                    jsd_vs_overall=jsd(Pd[d], Pall),
                    jsd_null_mean=null_j[:, d].mean(),
                    jsd_null_max=null_j[:, d].max(),
                    usage_entropy_bits=H2(Pd[d]),
                    effective_experts=2 ** H2(Pd[d]),
                    token_router_entropy_bits=ent_tok[m].mean(),
                    top8_router_mass=topw[l][m].sum(1).mean(),
                    isolation_expected_share=np.nansum(Pd[d] * Pbal[d]),
                    exclusive_mass_share_ge_0p5=Pd[d][Pbal[d] >= 0.5].sum(),
                    n_experts_share_ge_0p5=int((Pbal[d] >= 0.5).sum()),
                    top8_experts=" ".join(map(str, np.argsort(-Pd[d])[:8])),
                )
            )
        for a, b in itertools.combinations(range(ND), 2):
            pair_rows.append(
                dict(
                    layer=l,
                    dom_a=doms[a],
                    dom_b=doms[b],
                    jsd=jsd(Pd[a], Pd[b]),
                    jsd_null_mean_allpairs=np.mean(null_pair),
                    top8_jaccard=len(top8[a] & top8[b]) / len(top8[a] | top8[b]),
                    top8_overlap=len(top8[a] & top8[b]),
                )
            )
        SI = 1 - H2(Pbal.T) / np.log2(ND)
        for e in range(E):
            exp_rows.append(
                dict(
                    layer=l,
                    expert=e,
                    usage_share=Pall[e],
                    spec_index=SI[e],
                    spec_index_null_layer_mean=np.mean(null_si),
                    max_share=Pbal[:, e].max(),
                    max_over_uniform=Pbal[:, e].max() * ND,
                    top_domain=doms[Pbal[:, e].argmax()] if Pall[e] else "",
                    **{f"P({d}|e)": Pbal[i, e] for i, d in enumerate(doms)},
                    **{
                        f"frac_{d}_tokens_routed": C[i, e] / (dom == i).sum()
                        for i, d in enumerate(doms)
                    },
                )
            )
    dl = pd.DataFrame(rows)
    dp = pd.DataFrame(pair_rows)
    de = pd.DataFrame(exp_rows)
    dl.to_csv("results/layer_domain_metrics.csv", index=False)
    dp.to_csv("results/domain_pair_metrics.csv", index=False)
    de.to_csv("results/expert_domain_distribution.csv", index=False)
    for l in range(L):
        pd.DataFrame(Cs[l], index=doms).to_csv(
            f"results/counts/layer{l:02d}_domain_x_expert_counts.csv"
        ) if os.makedirs("results/counts", exist_ok=True) is None else None
    # ---------- orthogonality
    orows = []
    hist = {}
    iu = np.triu_indices(E, 1)
    for l in range(L):
        for nm in ("gate", "up", "down"):
            for k in ("real", "random"):
                c = O[f"w_{l}_{nm}_{k}"][iu]
                hist.setdefault(("w", nm, k), []).append(c)
                orows.append(
                    dict(
                        layer=l,
                        space="weight",
                        matrix=nm,
                        kind=k,
                        mean_abs_cos=np.abs(c).mean(),
                        max_abs_cos=np.abs(c).max(),
                        mean_signed_cos=c.mean(),
                        std_cos=c.std(),
                    )
                )
        for k in ("real", "random"):
            s = O[f"a_{l}_{k}_signed"][iu]
            a = O[f"a_{l}_{k}_abs"][iu]
            hist.setdefault(("a", k), []).append(s)
            hist.setdefault(("aabs", k), []).append(a)
            orows.append(
                dict(
                    layer=l,
                    space="activation",
                    matrix="expert_output",
                    kind=k,
                    mean_abs_cos=a.mean(),
                    max_abs_cos=a.max(),
                    mean_signed_cos=s.mean(),
                    std_cos=s.std(),
                    max_signed_cos=s.max(),
                    min_signed_cos=s.min(),
                )
            )
        # does output-similarity relate to co-routing?
        co = np.zeros((E, E))
        for kk in range(K):
            for jj in range(K):
                if kk != jj:
                    np.add.at(co, (topi[l][:, kk], topi[l][:, jj]), 1)
        s = O[f"a_{l}_real_signed"][iu]
        orows[-2]["corr_actcos_vs_coroute"] = np.corrcoef(s, co[iu])[0, 1]
    do = pd.DataFrame(orows)
    do.to_csv("results/orthogonality_by_layer.csv", index=False)
    # ---------- token-level / specialists
    cats = [
        "newline/space",
        "punct",
        "digit",
        "py_keyword",
        "latex/math_sym",
        "ascii_word",
        "non_ascii",
    ]
    kw = (
        set(keyword.kwlist)
        - {"in", "is", "and", "or", "not", "for", "with", "as", "if", "from", "at", "by"}
    ) | {
        "self",
        "print",
        "len",
        "range",
        "int",
        "str",
        "list",
        "dict",
        "append",
    }  # English-ambiguous keywords excluded

    def cat(s):
        st = s.strip()
        if st == "":
            return "newline/space"
        if any(ord(ch) > 127 for ch in st):
            return "non_ascii"
        if st in kw:
            return "py_keyword"
        if re.fullmatch(r"\d+", st):
            return "digit"
        if re.fullmatch(r"[\\\$\^_{}=+\-*/<>|]+|\\[a-zA-Z]+", st):
            return "latex/math_sym"
        if re.fullmatch(r"[^\w\s]+", st):
            return "punct"
        return "ascii_word"

    strs = {t: tok.decode([t]) for t in np.unique(tok_ids)}
    tcat = np.array([cats.index(cat(strs[t])) for t in tok_ids])
    base_cat = np.bincount(tcat, minlength=len(cats)) / N
    spec_rows = []
    for l in range(L):
        sub = de[de.layer == l]
        for d in doms:
            cand = (
                sub[sub[f"frac_{d}_tokens_routed"] >= 0.05]
                .sort_values(f"P({d}|e)", ascending=False)
                .head(3)
            )
            for rank, (_, r) in enumerate(cand.iterrows()):
                e = int(r.expert)
                m = (topi[l] == e).any(1)
                ti = tok_ids[m]
                vc = pd.Series(ti).value_counts()
                tot = m.sum()
                allc = pd.Series(tok_ids).value_counts()
                ent = H2((vc / tot).values)
                cover50 = int(np.searchsorted(np.cumsum(vc.values) / tot, 0.5) + 1)
                cd = np.bincount(tcat[m], minlength=len(cats)) / tot
                top = [f"{strs[t]!r}:{c}" for t, c in vc.head(12).items()]
                lift = (vc / tot) / (allc[vc.index] / N)
                lift = lift[vc >= 5].sort_values(ascending=False).head(8)
                # fraction of tokens of each top-type that go to this expert (token-determinism)
                det = np.mean(
                    [(m & (tok_ids == t)).sum() / (tok_ids == t).sum() for t in vc.head(10).index]
                )
                spec_rows.append(
                    dict(
                        layer=l,
                        domain=d,
                        rank=rank + 1,
                        expert=e,
                        share_balanced=r[f"P({d}|e)"],
                        frac_domain_tokens_routed=r[f"frac_{d}_tokens_routed"],
                        n_tokens=int(tot),
                        distinct_types=len(vc),
                        type_entropy_bits=ent,
                        types_for_50pct=cover50,
                        top_tokens=" | ".join(top),
                        top_lift_tokens=" | ".join(f"{strs[t]!r}x{v:.1f}" for t, v in lift.items()),
                        mean_routing_rate_of_top10_types=det,
                        **{f"cat_{c}": cd[i] for i, c in enumerate(cats)},
                    )
                )
    ds = pd.DataFrame(spec_rows)
    ds.to_csv("results/specialists_by_layer_domain.csv", index=False)
    pd.DataFrame([dict(category=c, corpus_share=base_cat[i]) for i, c in enumerate(cats)]).to_csv(
        "results/token_category_baseline.csv", index=False
    )
    # token-identity vs context: Jaccard of top-8 sets for same token type within domain / across domains / random pairs
    jrows = []
    types = [t for t, c in pd.Series(tok_ids).value_counts().items() if c >= 20]
    for l in range(L):
        S = topi[l]

        def jac(i, j, S=S):
            return len(set(S[i]) & set(S[j])) / len(set(S[i]) | set(S[j]))

        same_in = []
        same_x = []
        rand = []
        for t in types:
            idx = np.where(tok_ids == t)[0]
            for _ in range(10):
                i, j = rng.choice(idx, 2, replace=False)
                (same_in if dom[i] == dom[j] else same_x).append(jac(i, j))
        for _ in range(5000):
            i, j = rng.choice(N, 2, replace=False)
            rand.append(jac(i, j))
        jrows.append(
            dict(
                layer=l,
                same_token_same_domain=np.mean(same_in),
                same_token_cross_domain=np.mean(same_x),
                random_token_pairs=np.mean(rand),
                n_types=len(types),
            )
        )
    dj = pd.DataFrame(jrows)
    dj.to_csv("results/token_vs_context_routing_jaccard.csv", index=False)
    # ---------- plots
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in doms:
        s = dl[dl.domain == d]
        ax.plot(s.layer, s.jsd_vs_overall, marker="o", label=d)
    ax.plot(
        range(L),
        dl.groupby("layer").jsd_null_max.max(),
        "k--",
        label="permutation null (max)",
    )
    ax.set_xlabel("layer")
    ax.set_ylabel("JSD(P(e|domain) || P(e)) [bits]")
    ax.set_title("OLMoE-1B-7B-0924: routing divergence by layer")
    ax.legend()
    fig.tight_layout()
    fig.savefig("plots/jsd_by_layer.png", dpi=130)
    for l in dict.fromkeys((0, min(7, L - 1), L - 1)):
        Pd = Cs[l] / Cs[l].sum(1, keepdims=True)
        lift = ratio(Pd, Pd.mean(0), empty=1)
        order = np.lexsort((-lift.max(0), lift.argmax(0)))
        fig, ax = plt.subplots(figsize=(16, 3.6))
        im = ax.imshow(
            np.log2(np.clip(lift[:, order], 1 / 8, 8)),
            aspect="auto",
            cmap="RdBu_r",
            vmin=-3,
            vmax=3,
        )
        ax.set_yticks(range(ND))
        ax.set_yticklabels(doms)
        ax.set_xticks(range(E))
        ax.set_xticklabels(order, fontsize=6, rotation=90)
        ax.set_xlabel("expert id (sorted by preferred domain)")
        ax.set_title(f"layer {l}: log2 lift = log2[P(e|domain)/mean_d P(e|d)]")
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(f"plots/expert_domain_heatmap_layer{l:02d}.png", dpi=130)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, nm in zip(axs[:2], ("down", "up"), strict=True):
        r = np.concatenate(hist[("w", nm, "real")])
        b = np.concatenate(hist[("w", nm, "random")])
        bins = np.linspace(min(r.min(), b.min()), max(r.max(), b.max()), 120)
        ax.hist(b, bins, alpha=0.6, label="Gaussian same shape", density=True)
        ax.hist(r, bins, alpha=0.6, label="real experts", density=True)
        ax.set_title(f"weight-space cos ({nm}_proj, all layers, {len(r)} pairs)")
        ax.set_yscale("log")
        ax.legend()
    r = np.concatenate(hist[("a", "real")])
    b = np.concatenate(hist[("a", "random")])
    bins = np.linspace(-0.3, 1, 130)
    axs[2].hist(b, bins, alpha=0.6, density=True, label="random-weight experts")
    axs[2].hist(r, bins, alpha=0.6, density=True, label="real experts")
    axs[2].set_title("activation-space cos (token-avg, all layers)")
    axs[2].legend()
    axs[2].set_yscale("log")
    fig.tight_layout()
    fig.savefig("plots/orthogonality_hist.png", dpi=130)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    a = do[do.space == "activation"]
    for k in ("real", "random"):
        s = a[a.kind == k]
        ax.plot(s.layer, s.mean_signed_cos, marker="o", label=f"{k} mean signed")
        ax.plot(s.layer, s.mean_abs_cos, ls="--", label=f"{k} mean |cos|")
    ax.set_xlabel("layer")
    ax.set_ylabel("cos")
    ax.set_title("activation-space expert-output cosine by layer")
    ax.legend()
    fig.tight_layout()
    fig.savefig("plots/activation_cos_by_layer.png", dpi=130)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for d in doms:
        s = dl[dl.domain == d]
        ax.plot(s.layer, s.token_router_entropy_bits, marker="o", label=d)
    ax.axhline(6, color="k", ls=":", label="uniform (6 bits)")
    ax.set_xlabel("layer")
    ax.set_ylabel("per-token router entropy [bits]")
    ax.legend()
    fig.tight_layout()
    fig.savefig("plots/router_entropy_by_layer.png", dpi=130)
    print("ok")
