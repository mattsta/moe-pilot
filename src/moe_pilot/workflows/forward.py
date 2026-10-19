"""Forward pilot workflow."""

import torch


@torch.inference_mode()
def main(model_dir="models/olmoe", max_tokens=64, layers=None):
    # Layer-streaming OLMoE forward on CPU: loads one decoder layer at a time from safetensors,
    # logs router top-k per token per layer, computes weight-space & activation-space orthogonality.
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F

    from moe_pilot.artifacts import read_json
    from moe_pilot.engine import Engine
    from moe_pilot.model import rope_theta

    torch.manual_seed(0)
    np.random.seed(0)
    MAXTOK = max_tokens
    NLAY_RUN = layers
    engine = Engine(model_dir)
    cfg = engine.cfg
    eps = cfg.rms_norm_eps
    L = NLAY_RUN or cfg.num_hidden_layers
    E = cfg.num_experts
    K = cfg.num_experts_per_tok
    H = cfg.hidden_size
    NH = cfg.num_attention_heads
    HD = H // NH
    get = engine.get
    tok = engine.tok
    if cfg.num_hidden_layers < L:
        raise ValueError("Requested layers exceed model depth")
    corpus = read_json("corpus.json")
    ids = [tok(c["text"]).input_ids[:MAXTOK] for c in corpus]
    engine._validate(engine.batch(ids, MAXTOK), {}, 1)
    doms = sorted(set(c["domain"] for c in corpus))
    tok_ids = np.concatenate(ids)
    doc = np.concatenate([[i] * len(x) for i, x in enumerate(ids)])
    pos = np.concatenate([np.arange(len(x)) for x in ids])
    dom = np.array([doms.index(corpus[d]["domain"]) for d in doc])
    N = len(tok_ids)
    print("tokens", N, {d: int((dom == i).sum()) for i, d in enumerate(doms)}, flush=True)
    offs = np.cumsum([0] + [len(x) for x in ids])

    def rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

    inv = 1.0 / (rope_theta(cfg) ** (torch.arange(0, HD, 2).float() / HD))
    fr = torch.outer(torch.arange(MAXTOK).float(), inv)
    emb = torch.cat([fr, fr], -1)
    COS, SIN = emb.cos(), emb.sin()

    def rot(x):
        return torch.cat([-x[..., HD // 2 :], x[..., : HD // 2]], -1)

    # activation-probe sample: 100 tokens per domain
    rng = np.random.default_rng(0)
    probe = np.concatenate(
        [
            rng.choice(np.where(dom == i)[0], min(100, int((dom == i).sum())), replace=False)
            for i in range(len(doms))
        ]
    )
    h = get("model.embed_tokens.weight")[torch.from_numpy(tok_ids)]
    topi = np.zeros((L, N, K), np.int8)
    topw = np.zeros((L, N, K), np.float16)
    probs = np.zeros((L, N, E), np.float16)
    wcos = {}
    acos = {}
    XP = []

    def pair_cos(M):  # M: [E, D] -> ExE cos
        M = F.normalize(M, dim=1)
        return M @ M.T

    for l in range(L):
        t0 = time.time()
        p = f"model.layers.{l}."
        # ---- attention (per document, causal)
        x = rms(h, get(p + "input_layernorm.weight"))
        q = rms(x @ get(p + "self_attn.q_proj.weight").T, get(p + "self_attn.q_norm.weight"))
        k = rms(x @ get(p + "self_attn.k_proj.weight").T, get(p + "self_attn.k_norm.weight"))
        v = x @ get(p + "self_attn.v_proj.weight").T
        if cfg.clip_qkv is not None:
            q, k, v = [z.clamp(-cfg.clip_qkv, cfg.clip_qkv) for z in (q, k, v)]
        att = torch.empty_like(x)
        for d in range(len(ids)):
            a, b = offs[d], offs[d + 1]
            T = b - a
            qq, kk, vv = [z[a:b].view(T, NH, HD).transpose(0, 1) for z in (q, k, v)]
            c, s = COS[:T], SIN[:T]
            qq = qq * c + rot(qq) * s
            kk = kk * c + rot(kk) * s
            att[a:b] = (
                F.scaled_dot_product_attention(qq[None], kk[None], vv[None], is_causal=True)[0]
                .transpose(0, 1)
                .reshape(T, H)
            )
        h = h + att @ get(p + "self_attn.o_proj.weight").T
        # ---- MoE
        x = rms(h, get(p + "post_attention_layernorm.weight"))
        pr = torch.softmax(x @ get(p + "mlp.gate.weight").T, -1)
        tw, ti = torch.topk(pr, K, -1)  # norm_topk_prob=False
        if cfg.norm_topk_prob:
            tw = tw / tw.sum(-1, keepdim=True)
        probs[l] = pr.numpy()
        topi[l] = ti.numpy()
        topw[l] = tw.numpy()
        G = engine.store.expert_weights(l, "gate")  # E,I,H
        U = engine.store.expert_weights(l, "up")
        D = engine.store.expert_weights(l, "down")  # E,H,I
        out = torch.zeros_like(h)
        for e in range(E):
            r, c = torch.where(ti == e)
            if len(r) == 0:
                continue
            xe = x[r]
            ye = (F.silu(xe @ G[e].T) * (xe @ U[e].T)) @ D[e].T
            out.index_add_(0, r, ye * tw[r, c, None])
        # ---- weight-space orthogonality (real vs Gaussian same shape)
        for nm, W in (("gate", G), ("up", U), ("down", D)):
            wcos[(l, nm, "real")] = pair_cos(W.reshape(E, -1)).numpy()
            R = torch.randn_like(W)
            wcos[(l, nm, "random")] = pair_cos(R.reshape(E, -1)).numpy()
            del R
        # ---- activation-space orthogonality on probe tokens
        xp = x[torch.from_numpy(probe)]
        XP.append(xp.numpy().copy())

        def run_all(G, U, D, xp=xp):
            return torch.stack(
                [(F.silu(xp @ G[e].T) * (xp @ U[e].T)) @ D[e].T for e in range(E)]
            )  # E,P,H

        def act_cos(Y):
            Yn = F.normalize(Y, dim=-1)
            C = torch.einsum("eph,fph->efp", Yn, Yn)  # E,E,P
            return C.mean(-1).numpy(), C.abs().mean(-1).numpy()

        acos[(l, "real")] = act_cos(run_all(G, U, D))

        def gauss_like(W):
            return (
                torch.randn_like(W)
                * (W.flatten(1).norm(dim=1) / np.sqrt(W[0].numel()))[:, None, None]
            )

        acos[(l, "random")] = act_cos(run_all(gauss_like(G), gauss_like(U), gauss_like(D)))
        h = h + out
        del G, U, D
        print(f"layer {l} done {time.time() - t0:.1f}s", flush=True)
    # sanity: LM loss (next-token) if full depth
    if cfg.num_hidden_layers == L:
        hn = rms(h, get("model.norm.weight"))
        losses = []
        Wl = get("lm_head.weight")
        for d in range(len(ids)):
            a, b = offs[d], offs[d + 1]
            lg = hn[a : b - 1] @ Wl.T
            losses.append(
                F.cross_entropy(lg, torch.from_numpy(tok_ids[a + 1 : b]).long(), reduction="none")
            )
        ls = torch.cat(losses)
        dd = np.concatenate([[dom[offs[d]]] * (offs[d + 1] - offs[d] - 1) for d in range(len(ids))])
        print(
            "LM loss overall",
            ls.mean().item(),
            {doms[i]: ls[torch.from_numpy(dd == i)].mean().item() for i in range(len(doms))},
        )
    np.savez_compressed(
        "routing.npz",
        topi=topi,
        topw=topw,
        probs=probs,
        tok_ids=tok_ids,
        doc=doc,
        pos=pos,
        dom=dom,
        doms=np.array(doms),
        probe=probe,
    )
    np.savez_compressed(
        "ortho_raw.npz",
        **{f"w_{l}_{nm}_{k}": v for (l, nm, k), v in wcos.items()},
        **{
            f"a_{l}_{k}_{s}": v[j]
            for (l, k), v in acos.items()
            for j, s in enumerate(("signed", "abs"))
        },
    )
    torch.save(h[:5].clone(), "h_check.pt")
    np.save("probe_moe_inputs.npy", np.stack(XP))
    print("SAVED")
