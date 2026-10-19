"""Ablate analyze pilot workflow."""


def main(model_dir="models/olmoe"):
    import json
    import os

    import matplotlib
    import numpy as np
    import pandas as pd

    from moe_pilot.artifacts import read_json

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT = "results/ablations"
    doms = ["code", "math", "multilingual", "prose", "science"]
    meta = pd.read_csv(f"{OUT}/conditions.csv")
    conds = read_json(f"{OUT}/conditions.json")
    valid = np.load(f"{OUT}/valid.npy")
    dom = np.load(f"{OUT}/token_domain.npy")
    doc = np.load(f"{OUT}/token_doc.npy")
    base = np.load(f"{OUT}/losses/base.npy")
    # expand aliases (conditions with identical masks were computed once under a canonical name)
    import ast
    import re

    SD = {
        "L7_E45": "code",
        "L7_E35": "math",
        "L7_E27": "multilingual",
        "L7_E47": "prose",
        "L7_E63": "science",
        "L15_E8": "code",
        "L15_E30": "code",
        "L15_E3": "math",
        "L15_E12": "multilingual",
        "L15_E0": "multilingual",
        "L15_E25": "prose",
        "L15_E59": "prose",
        "L15_E38": "science",
        "L0_E21": "code",
        "L0_E9": "science",
    }

    def parse(n):
        seed = int(re.search(r"_rand(\d)$", n).group(1)) if "_rand" in n else -1
        b = re.sub(r"_rand\d$", "", n)
        if b.startswith("single_"):
            _, l, e = b.split("_")
            return dict(
                kind="single",
                domain=SD[f"{l}_{e}"],
                scope=l,
                k=1,
                control=seed >= 0,
                seed=seed,
            )
        _, d, k, sc = b.split("_")
        return dict(kind="joint", domain=d, scope=sc, k=int(k[1:]), control=seed >= 0, seed=seed)

    FILE = {}
    rows = []
    for _, r in meta.iterrows():
        names = [r["name"]] + (ast.literal_eval(r.aliases) if isinstance(r.aliases, str) else [])
        for n in names:
            FILE[n] = r["name"]
            rows.append(dict(name=n, canonical=r["name"], **parse(n)))
    meta = pd.DataFrame(rows)
    meta = meta[meta.canonical.map(lambda c: os.path.exists(f"{OUT}/losses/{c}.npy"))].copy()
    meta.to_csv(f"{OUT}/conditions_expanded.csv", index=False)
    conds = {n: conds[FILE[n]] for n in meta.name}
    # on-domain routing mass removed (analysis corpus, unablated routing), per eval domain
    R = np.load("routing.npz")
    topi = R["topi"].astype(int)
    topw = R["topw"].astype(np.float32)
    rdom = R["dom"]
    rdoms = list(R["doms"])
    mass = np.zeros((16, len(rdoms), 64))
    for l in range(16):
        for i in range(len(rdoms)):
            m = rdom == i
            np.add.at(mass[l, i], topi[l][m].ravel(), topw[l][m].ravel())
            mass[l, i] /= m.sum()

    def rmass(n, d):
        return sum(mass[int(l), rdoms.index(d), v].sum() for l, v in conds[n].items())

    ndoc = doc.max() + 1

    def per_doc(x):  # mean over valid tokens per doc
        s = np.bincount(doc[valid], x[valid], minlength=ndoc)
        c = np.bincount(doc[valid], minlength=ndoc)
        return s / np.maximum(c, 1)

    docdom = np.array([dom[doc == i][0] for i in range(ndoc)])

    def boot_ci(v, nb=2000, seed=0):
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(v), (nb, len(v)))
        b = v[idx].mean(1)
        return np.percentile(b, 2.5), np.percentile(b, 97.5)

    D = {n: per_doc(np.load(f"{OUT}/losses/{FILE[n]}.npy") - base) for n in meta.name}
    rows = []
    for _, r in meta[~meta.control].iterrows():
        n = r["name"]
        ctr = [f"{n}_rand{s}" for s in range(3) if f"{n}_rand{s}" in D]
        row = dict(
            condition=n,
            kind=r.kind,
            spec_domain=r.domain,
            scope=r.scope,
            k=r.k,
            n_controls=len(ctr),
            masked=json.dumps(conds[n]) if r.scope != "all" else f"top-{r.k} per layer x16",
            computed_as=FILE[n],
        )
        for d in doms:
            m = docdom == doms.index(d)
            row[f"d_{d}"] = D[n][m].mean()
            if ctr:
                row[f"ctrl_{d}"] = np.mean([D[c][m].mean() for c in ctr])
            row[f"rmass_{d}"] = rmass(n, d)
            if ctr:
                row[f"ctrl_rmass_{d}"] = np.mean([rmass(c, d) for c in ctr])
        on = doms.index(r.domain)
        m_on = docdom == on
        off = [d for d in doms if d != r.domain]
        row["on_delta"] = row[f"d_{r.domain}"]
        row["off_delta_mean"] = np.mean([row[f"d_{d}"] for d in off])
        row["off_delta_max"] = max(row[f"d_{d}"] for d in off)
        row["selectivity_ratio"] = (
            row["on_delta"] / row["off_delta_mean"] if row["off_delta_mean"] > 0 else np.nan
        )
        row["on_minus_off"] = row["on_delta"] - row["off_delta_mean"]
        sd = sum(map(ord, FILE[n]))
        lo, hi = boot_ci(D[n][m_on], seed=sd)
        row["on_delta_ci_lo"], row["on_delta_ci_hi"] = lo, hi
        # paired doc-level: specialist minus control mean (on-domain), and on-off selectivity of controls
        if ctr:
            diff = D[n][m_on] - np.mean([D[c][m_on] for c in ctr], 0)
            lo, hi = boot_ci(diff, seed=sd + 1)
            row["spec_minus_ctrl_on"] = diff.mean()
            row["spec_minus_ctrl_ci_lo"], row["spec_minus_ctrl_ci_hi"] = lo, hi
            row["ctrl_on_max"] = max(D[c][m_on].mean() for c in ctr)
            row["ctrl_on_minus_off"] = row[f"ctrl_{r.domain}"] - np.mean(
                [row[f"ctrl_{d}"] for d in off]
            )
            nec = (
                row["on_delta_ci_lo"] > 0
                and row["spec_minus_ctrl_ci_lo"] > 0
                and row["on_delta"] > row["ctrl_on_max"]
            )
            row["verdict"] = (
                "necessary(>controls)"
                if nec
                else (
                    "redundant(<=controls/ns)"
                    if row["spec_minus_ctrl_ci_lo"] <= 0
                    else "harmful-but-ns"
                )
            )
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/ablation_summary.csv", index=False)
    mat = res.set_index("condition")[[f"d_{d}" for d in doms]]
    mat.columns = doms
    mat.to_csv(f"{OUT}/delta_loss_matrix_specialists.csv")
    cm = meta[meta.control].name
    cmat = pd.DataFrame(
        {d: [D[n][docdom == doms.index(d)].mean() for n in cm] for d in doms}, index=cm
    )
    cmat.to_csv(f"{OUT}/delta_loss_matrix_controls.csv")
    print(
        "base loss per domain:",
        {d: float(base[valid & (dom == i)].mean()) for i, d in enumerate(doms)},
    )
    # plots
    os.makedirs("plots/ablations", exist_ok=True)
    for kind in ("single", "joint"):
        s = res[res.kind == kind]
        if s.empty:
            continue
        fig, axs = plt.subplots(
            1, 2, figsize=(14, max(3, 0.32 * len(s))), gridspec_kw={"width_ratios": [1, 1]}
        )
        A = s[[f"d_{d}" for d in doms]].values
        Cc = s[[f"ctrl_{d}" for d in doms]].values if f"ctrl_{doms[0]}" in s else np.zeros_like(A)
        v = np.nanmax(np.abs(np.r_[A.ravel(), Cc.ravel()])) if len(A) else 1
        for ax, M, t in (
            (axs[0], A, "specialist ablation"),
            (axs[1], Cc, "mass-matched random (mean of seeds)"),
        ):
            im = ax.imshow(M, cmap="magma_r", vmin=0, vmax=v, aspect="auto")
            ax.set_xticks(range(5))
            ax.set_xticklabels(doms)
            ax.set_yticks(range(len(s)))
            ax.set_yticklabels(s.condition, fontsize=7)
            ax.set_title(f"{t}: Δ loss (nats/token)")
            for i in range(M.shape[0]):
                for j in range(5):
                    ax.text(
                        j,
                        i,
                        f"{M[i, j]:.3f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="cyan",
                    )
        plt.colorbar(im, ax=axs[1])
        fig.tight_layout()
        fig.savefig(f"plots/ablations/delta_loss_{kind}.png", dpi=130)
    s = res[res.kind == "joint"]
    if not s.empty:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
        for ax, sc in zip(axs, ("L7", "L15", "all"), strict=True):
            for d in doms:
                q = s[(s.scope == sc) & (s.spec_domain == d)].sort_values("k")
                if q.empty:
                    continue
                ax.plot(q.k, q.on_delta, marker="o", label=f"{d} specialists")
                ax.plot(q.k, q[f"ctrl_{d}"], ls="--", color=ax.lines[-1].get_color())
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 4, 8])
            ax.set_xticklabels([1, 4, 8])
            ax.set_title(f"scope {sc}: on-domain Δloss (dashed = matched random)")
            ax.set_xlabel("k experts masked per layer")
            ax.set_yscale("symlog", linthresh=0.01)
        axs[0].legend(fontsize=7)
        axs[0].set_ylabel("Δ loss (nats/token)")
        fig.tight_layout()
        fig.savefig("plots/ablations/on_domain_delta_vs_k.png", dpi=130)
    print("ok", len(res), "specialist conditions,", len(cm), "controls")
    # ---- second control family: non-specialists matched on ON-DOMAIN routing mass (ablate_ondomain_ctrl.py)
    if os.path.exists(f"{OUT}/conditions_ondomain_ctrl.csv"):
        oc = pd.read_csv(f"{OUT}/conditions_ondomain_ctrl.csv")
        oc = oc[oc.name.map(lambda n: os.path.exists(f"{OUT}/losses/{n}.npy"))]
        DO = {n: per_doc(np.load(f"{OUT}/losses/{n}.npy") - base) for n in oc.name}
        rows = []
        for parent, g in oc.groupby("parent"):
            names = [parent] + list(
                meta[(meta.canonical == parent) & (~meta.control)].name.drop_duplicates()
            )
            d = g.domain.iloc[0]
            m_on = docdom == doms.index(d)
            off = [x for x in doms if x != d]
            for n in dict.fromkeys(names):
                if n not in D:
                    continue
                cd = {
                    x: np.mean([DO[c][docdom == doms.index(x)].mean() for c in g.name])
                    for x in doms
                }
                diff = D[n][m_on] - np.mean([DO[c][m_on] for c in g.name], 0)
                lo, hi = boot_ci(diff, seed=sum(map(ord, parent)) + 7)
                spec_on = D[n][m_on].mean()
                rows.append(
                    dict(
                        condition=n,
                        domain=d,
                        n_odctrl=len(g),
                        target_on_mass=g.target_on_mass.iloc[0],
                        odctrl_on_mass_mean=g.ctrl_on_mass.mean(),
                        spec_on_delta=spec_on,
                        odctrl_on_delta=cd[d],
                        odctrl_off_delta_mean=np.mean([cd[x] for x in off]),
                        odctrl_max_on=max(DO[c][m_on].mean() for c in g.name),
                        spec_minus_odctrl=diff.mean(),
                        ci_lo=lo,
                        ci_hi=hi,
                        ratio_spec_over_odctrl=spec_on / cd[d] if cd[d] > 0 else np.nan,
                        verdict_vs_odctrl=(
                            "specialist>on-mass-matched"
                            if lo > 0
                            else ("specialist<on-mass-matched" if hi < 0 else "not distinguishable")
                        ),
                        **{f"odctrl_{x}": cd[x] for x in doms},
                    )
                )
        pd.DataFrame(rows).to_csv(f"{OUT}/ablation_vs_ondomain_mass_controls.csv", index=False)
        print("odctrl rows", len(rows))
