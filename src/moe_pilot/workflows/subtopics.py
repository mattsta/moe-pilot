"""Subtopics pilot workflow."""


def main(model_dir="models/olmoe"):
    import itertools
    import os

    import matplotlib
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from moe_pilot.artifacts import read_json
    from moe_pilot.metrics import entropy_bits as H2
    from moe_pilot.metrics import jensen_shannon as jsd
    from moe_pilot.metrics import ratio

    OUT = "results/subtopics"
    os.makedirs("plots/subtopics", exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    C = read_json("subtopic_corpus.json")
    if not os.path.exists(f"{OUT}/routing.npz"):
        from moe_pilot.engine import Engine

        M = Engine(model_dir)

        ids = [M.tok(x["text"]).input_ids for x in C]
        B = M.batch(ids, 64)
        res, valid, rl = M.run(B, {}, log_routing=True)
        np.savez_compressed(
            f"{OUT}/routing.npz",
            topi=np.stack([r[0] for r in rl]),
            topw=np.stack([r[1] for r in rl]),
            ent=np.stack(
                [
                    -(
                        r[2].astype(np.float32)
                        * np.log2(np.clip(r[2].astype(np.float32), 1e-12, 1))
                    ).sum(1)
                    for r in rl
                ]
            ),
            tok=B.tok.numpy(),
            doc=B.doc,
            loss=res["base"],
            valid=valid,
        )
    R = np.load(f"{OUT}/routing.npz")
    topi = R["topi"].astype(int)
    doc = R["doc"]
    tok = R["tok"]
    L, N, K = topi.shape
    E = 64
    subs = list(dict.fromkeys(f"{c['domain']}/{c['subtopic']}" for c in C))
    S = len(subs)
    sub_of_doc = np.array([subs.index(f"{c['domain']}/{c['subtopic']}") for c in C])
    sub = sub_of_doc[doc]
    doms = list(dict.fromkeys(s.split("/")[0] for s in subs))
    dom_of_sub = np.array([doms.index(s.split("/")[0]) for s in subs])

    def P_of(l, lab):
        Cn = np.stack(
            [np.bincount(topi[l][lab == s].ravel(), minlength=E) for s in range(S)]
        ).astype(float)
        return Cn / Cn.sum(1, keepdims=True), Cn

    rng = np.random.default_rng(0)
    NP = 20

    def perm_within_domain():
        p = sub_of_doc.copy()
        for d in range(len(doms)):
            idx = np.where(dom_of_sub[sub_of_doc] == d)[0]
            p[idx] = rng.permutation(sub_of_doc[idx])
        return p[doc]

    perms_w = [perm_within_domain() for _ in range(NP)]
    perms_g = [rng.permutation(sub_of_doc)[doc] for _ in range(NP)]
    pairs = list(itertools.combinations(range(S), 2))
    within = [(a, b) for a, b in pairs if dom_of_sub[a] == dom_of_sub[b]]
    between = [(a, b) for a, b in pairs if dom_of_sub[a] != dom_of_sub[b]]
    lay = []
    pairrows = []
    exprows = []
    Ps = {}

    def si_within(
        P, d
    ):  # per-expert subtopic specialization within domain d (balanced over its subtopics)
        ss = np.where(dom_of_sub == d)[0]
        Q = P[ss]
        Pb = ratio(Q, Q.sum(0, keepdims=True))
        return 1 - H2(Pb.T) / np.log2(len(ss)), Pb, ss

    for l in range(L):
        P, Cn = P_of(l, sub)
        Ps[l] = (P, Cn)
        Pw = [P_of(l, pl)[0] for pl in perms_w]
        Pg = [P_of(l, pl)[0] for pl in perms_g]
        for d, dn in enumerate(doms):
            wp = [(a, b) for a, b in within if dom_of_sub[a] == d]
            real = np.mean([jsd(P[a], P[b]) for a, b in wp])
            null = [np.mean([jsd(Q[a], Q[b]) for a, b in wp]) for Q in Pw]
            si, _, _ = si_within(P, d)
            sin = [np.nanmean(si_within(Q, d)[0]) for Q in Pw]
            use = P[dom_of_sub == d].mean(0)
            ok = ~np.isnan(si)
            sinw = [
                np.nansum(si_within(Q, d)[0] * use) / use[~np.isnan(si_within(Q, d)[0])].sum()
                for Q in Pw
            ]
            # pooled domain distribution & subtopic vs domain JSD
            ss = np.where(dom_of_sub == d)[0]
            pool = P[ss].mean(0)
            lay.append(
                dict(
                    layer=l,
                    domain=dn,
                    n_subtopics=len(ss),
                    within_domain_pair_jsd=real,
                    within_null_mean=np.mean(null),
                    within_null_max=np.max(null),
                    subtopic_vs_domain_jsd=np.mean([jsd(P[s], pool) for s in ss]),
                    subtopic_SI_mean=np.nanmean(si),
                    subtopic_SI_max=np.nanmax(si),
                    subtopic_SI_null_mean=np.mean(sin),
                    subtopic_SI_usage_weighted=(si[ok] * use[ok]).sum() / use[ok].sum(),
                    subtopic_SI_usage_weighted_null=np.mean(sinw),
                    n_experts_subSI_gt_null_max=int(
                        (si[ok] > max(np.nanmax(si_within(Q, d)[0]) for Q in Pw)).sum()
                    ),
                )
            )
            for e in range(E):
                exprows.append(
                    dict(
                        layer=l,
                        domain=dn,
                        expert=e,
                        subtopic_SI=si[e],
                        **{
                            f"P({subs[s]}|e,dom)": v
                            for s, v in zip(ss, si_within(P, d)[1][:, e], strict=True)
                        },
                        **{f"frac_{subs[s]}": Cn[s, e] / (sub == s).sum() for s in ss},
                    )
                )
        pooled = np.stack([P[dom_of_sub == d].mean(0) for d in range(len(doms))])
        bw = np.mean([jsd(P[a], P[b]) for a, b in between])
        bnull = np.mean([np.mean([jsd(Q[a], Q[b]) for a, b in between]) for Q in Pg])
        for r in lay[-len(doms) :]:
            r.update(
                between_domain_subtopic_pair_jsd=bw,
                between_null_global_shuffle=bnull,
                domain_pooled_pair_jsd_mean=np.mean(
                    [
                        jsd(pooled[a], pooled[b])
                        for a, b in itertools.combinations(range(len(doms)), 2)
                    ]
                ),
            )
        for a, b in pairs:
            pairrows.append(
                dict(
                    layer=l,
                    a=subs[a],
                    b=subs[b],
                    same_domain=dom_of_sub[a] == dom_of_sub[b],
                    jsd=jsd(P[a], P[b]),
                )
            )
    dl = pd.DataFrame(lay)
    dl.to_csv(f"{OUT}/layer_within_vs_between_jsd.csv", index=False)
    dp = pd.DataFrame(pairrows)
    dp.to_csv(f"{OUT}/subtopic_pair_jsd.csv", index=False)
    de = pd.DataFrame(exprows)
    de.to_csv(f"{OUT}/expert_subtopic_distribution.csv", index=False)
    # subtopic specialists (within domain), top-3 at each layer
    sp = []
    for l in range(L):
        for d, dn in enumerate(doms):
            x = de[(de.layer == l) & (de.domain == dn)]
            ss = np.where(dom_of_sub == d)[0]
            for s in ss:
                col = f"P({subs[s]}|e,dom)"
                c = x[x[f"frac_{subs[s]}"] >= 0.05].sort_values(col, ascending=False).head(3)
                for rk, (_, r) in enumerate(c.iterrows()):
                    m = (topi[l] == int(r.expert)).any(1) & (sub == s)
                    vc = pd.Series(tok[m]).value_counts().head(8)
                    sp.append(
                        dict(
                            layer=l,
                            subtopic=subs[s],
                            rank=rk + 1,
                            expert=int(r.expert),
                            share_within_domain=r[col],
                            uniform=1 / len(ss),
                            share_over_uniform=r[col] * len(ss),
                            frac_subtopic_tokens_routed=r[f"frac_{subs[s]}"],
                            subtopic_SI=r.subtopic_SI,
                            top_token_ids=" ".join(map(str, vc.index)),
                        )
                    )
    dsp = pd.DataFrame(sp)
    try:
        from transformers import AutoTokenizer

        T = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        dsp["top_tokens"] = [
            " | ".join(repr(T.decode([int(t)])) for t in s.split()) for s in dsp.top_token_ids
        ]
    except Exception as ex:
        print("tok decode failed", ex)
    dsp.to_csv(f"{OUT}/subtopic_specialists.csv", index=False)
    # do original domain specialists (from the 5-domain run) split by subtopic or stay shared?
    orig = pd.read_csv("results/expert_domain_distribution.csv")
    rows = []
    for l in range(L):
        for dn in doms:
            o = orig[orig.layer == l].sort_values(f"P({dn}|e)", ascending=False)
            o = o[o[f"frac_{dn}_tokens_routed"] >= 0.05].head(3)
            x = de[(de.layer == l) & (de.domain == dn)].set_index("expert")
            ss = np.where(dom_of_sub == doms.index(dn))[0]
            nullmax = max(
                np.nanmax(si_within(Q, doms.index(dn))[0])
                for Q in [P_of(l, pl)[0] for pl in perms_w[:5]]
            )
            for _, r in o.iterrows():
                e = int(r.expert)
                fr = {subs[s]: x.loc[e, f"frac_{subs[s]}"] for s in ss}
                rows.append(
                    dict(
                        layer=l,
                        domain=dn,
                        expert=e,
                        orig_domain_share=r[f"P({dn}|e)"],
                        subtopic_SI=x.loc[e, "subtopic_SI"],
                        subtopic_SI_null_max=nullmax,
                        top_subtopic=max(fr, key=fr.get),
                        min_frac=min(fr.values()),
                        max_frac=max(fr.values()),
                        **{f"frac_{k}": v for k, v in fr.items()},
                    )
                )
    pd.DataFrame(rows).to_csv(f"{OUT}/domain_specialists_by_subtopic.csv", index=False)
    # plots
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for dn in doms:
        s = dl[dl.domain == dn]
        axs[0].plot(s.layer, s.within_domain_pair_jsd, marker="o", label=f"within {dn}")
        axs[0].plot(s.layer, s.within_null_max, ls=":", color=axs[0].lines[-1].get_color())
    s = dl[dl.domain == doms[0]]
    axs[0].plot(
        s.layer,
        s.between_domain_subtopic_pair_jsd,
        "k-",
        lw=2.5,
        marker="s",
        label="between-domain subtopic pairs",
    )
    axs[0].plot(s.layer, s.between_null_global_shuffle, "k:", label="global shuffle null")
    axs[0].set_xlabel("layer")
    axs[0].set_ylabel("mean pairwise JSD [bits]")
    axs[0].legend(fontsize=7)
    axs[0].set_title("subtopic routing divergence (dotted = within-domain shuffle null max)")
    for dn in doms:
        s = dl[dl.domain == dn]
        axs[1].plot(s.layer, s.subtopic_SI_usage_weighted, marker="o", label=dn)
        axs[1].plot(
            s.layer,
            s.subtopic_SI_usage_weighted_null,
            ls=":",
            color=axs[1].lines[-1].get_color(),
        )
    axs[1].set_xlabel("layer")
    axs[1].set_ylabel("usage-weighted subtopic SI")
    axs[1].legend()
    axs[1].set_title("within-domain subtopic SI (dotted = shuffled)")
    fig.tight_layout()
    fig.savefig("plots/subtopics/jsd_within_vs_between.png", dpi=130)
    for l in (0, 7, 15):
        P = Ps[l][0]
        lift = np.clip(ratio(P, P.mean(0), empty=1), 1 / 8, 8)
        order = np.lexsort((-lift.max(0), lift.argmax(0)))
        fig, ax = plt.subplots(figsize=(16, 6))
        im = ax.imshow(np.log2(lift[:, order]), aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
        ax.set_yticks(range(S))
        ax.set_yticklabels(subs, fontsize=8)
        ax.set_xticks(range(E))
        ax.set_xticklabels(order, fontsize=6, rotation=90)
        ax.set_title(f"layer {l}: log2 lift of P(e|subtopic) over mean across subtopics")
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(f"plots/subtopics/subtopic_expert_heatmap_layer{l:02d}.png", dpi=130)
    fig, ax = plt.subplots(figsize=(9, 8))
    l = 15
    P = Ps[l][0]
    J = np.array([[jsd(P[a], P[b]) for b in range(S)] for a in range(S)])
    im = ax.imshow(J, cmap="viridis")
    ax.set_xticks(range(S))
    ax.set_xticklabels(subs, rotation=90, fontsize=7)
    ax.set_yticks(range(S))
    ax.set_yticklabels(subs, fontsize=7)
    ax.set_title("layer 15 pairwise JSD between subtopics [bits]")
    plt.colorbar(im)
    fig.tight_layout()
    fig.savefig("plots/subtopics/subtopic_jsd_matrix_layer15.png", dpi=130)
    print("ok")
