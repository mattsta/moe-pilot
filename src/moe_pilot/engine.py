"""Layer-streaming inference and routing ablations for OLMoE on PyTorch devices.

The engine owns its configuration and tensor store. Constructing a batch does
not tokenize or download data. All inference runs without autograd.
"""

import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import version
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from moe_pilot.checkpoints import Checkpoints
from moe_pilot.devices import DeviceSelection, resolve_device, to_numpy
from moe_pilot.model import ModelStore, rope_theta


@dataclass
class Batch:
    """Host-side token/document metadata with explicit right-padding indices."""

    ids: list[list[int]]
    T: int

    def __post_init__(self):
        if type(self.T) is not int or self.T < 2:
            raise ValueError("max_tokens must be at least 2 for next-token evaluation")
        self.ids = [list(document[: self.T]) for document in self.ids]
        if not self.ids or any(len(document) < 2 for document in self.ids):
            raise ValueError("Each batch must contain documents with at least two tokens")
        if any(
            not isinstance(token, Integral) or isinstance(token, bool)
            for document in self.ids
            for token in document
        ):
            raise ValueError("Token IDs must be integers")
        self.lens = np.array([len(document) for document in self.ids])
        self.N = int(self.lens.sum())
        self.tok = torch.tensor(np.concatenate(self.ids), dtype=torch.long, device="cpu")
        if (self.tok < 0).any():
            raise ValueError("Token IDs must be nonnegative")
        self.doc = np.repeat(np.arange(len(self.ids)), self.lens)
        self.pos = np.concatenate([np.arange(n) for n in self.lens])
        self.flat_idx = torch.from_numpy(self.doc * self.T + self.pos).long()


class Engine:
    """Device-aware reference engine for standard OLMoE checkpoints.

    ``model_dir`` must already contain config and safetensors. Tokenizer files
    are needed only when accessing ``tok``. Thread settings are left to the
    caller; constructing this library object does not change process settings.
    """

    def __init__(self, model_dir, *, device="auto"):
        self.store = ModelStore(model_dir)
        self.selection = device if isinstance(device, DeviceSelection) else resolve_device(device)
        self.device = self.selection.device
        self.cfg = self.store.config
        cfg = self.cfg
        if cfg.model_type != "olmoe":
            raise ValueError("Engine supports OLMoE checkpoints only")
        if cfg.num_key_value_heads != cfg.num_attention_heads or cfg.attention_bias:
            raise ValueError("Grouped-query attention and attention bias are unsupported")
        if cfg.hidden_act != "silu":
            raise ValueError("Engine supports the SiLU expert activation only")
        self.L, self.E, self.K = cfg.num_hidden_layers, cfg.num_experts, cfg.num_experts_per_tok
        if self.L < 1 or not 1 <= self.K <= self.E:
            raise ValueError("Model must have positive depth and a valid expert/top-k count")
        self.H, self.NH = cfg.hidden_size, cfg.num_attention_heads
        self.HD = self.H // self.NH
        if self.H % self.NH or self.HD % 2:
            raise ValueError("Attention head dimension must be even and divide hidden_size")
        self.theta = rope_theta(cfg)
        self._tokenizer = None

    @property
    def tok(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.store.directory, local_files_only=True
            )
        return self._tokenizer

    def batch(self, ids, max_tokens=64):
        return Batch(ids, max_tokens)

    def get(self, name):
        return self.store.get(name).to(self.device)

    def _embed(self, batch):
        # Select token rows while staging from disk; do not upload an entire
        # embedding table just to retain a small corpus's hidden states.
        return self.store.get("model.embed_tokens.weight")[batch.tok].to(self.device)

    def rms(self, x, weight):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.cfg.rms_norm_eps) * weight

    def load_layer(self, layer):
        prefix = f"model.layers.{layer}."
        weights = {
            name: self.get(prefix + name)
            for name in (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "mlp.gate.weight",
            )
        }
        packed = prefix + "mlp.experts.gate_up_proj"
        if packed in self.store.weight_map:
            gate, up = self.get(packed).chunk(2, dim=1)
            weights["G"], weights["U"] = list(gate.unbind()), list(up.unbind())
            weights["D"] = list(self.get(prefix + "mlp.experts.down_proj").unbind())
        else:
            for key, projection in (("G", "gate"), ("U", "up"), ("D", "down")):
                weights[key] = [
                    self.get(prefix + f"mlp.experts.{expert}.{projection}_proj.weight")
                    for expert in range(self.E)
                ]
        return weights

    def attn(self, hidden, weights, batch):
        x = self.rms(hidden, weights["input_layernorm.weight"])
        q = self.rms(x @ weights["self_attn.q_proj.weight"].T, weights["self_attn.q_norm.weight"])
        k = self.rms(x @ weights["self_attn.k_proj.weight"].T, weights["self_attn.k_norm.weight"])
        v = x @ weights["self_attn.v_proj.weight"].T
        if self.cfg.clip_qkv is not None:
            q, k, v = [z.clamp(-self.cfg.clip_qkv, self.cfg.clip_qkv) for z in (q, k, v)]
        inv = self.theta ** (-torch.arange(0, self.HD, 2, device=self.device).float() / self.HD)
        frequencies = torch.outer(torch.arange(batch.T, device=self.device).float(), inv)
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        cosine, sine = embeddings.cos(), embeddings.sin()

        def pad(z):
            padded = torch.zeros(
                len(batch.ids) * batch.T, self.H, device=self.device, dtype=hidden.dtype
            )
            padded[batch.flat_idx.to(self.device)] = z
            return padded.view(len(batch.ids), batch.T, self.NH, self.HD).transpose(1, 2)

        def rotate(z):
            return torch.cat((-z[..., self.HD // 2 :], z[..., : self.HD // 2]), dim=-1)

        q, k, v = pad(q), pad(k), pad(v)
        q, k = q * cosine + rotate(q) * sine, k * cosine + rotate(k) * sine
        output = torch.cat(
            [
                F.scaled_dot_product_attention(
                    q[i : i + 64], k[i : i + 64], v[i : i + 64], is_causal=True
                )
                for i in range(0, len(batch.ids), 64)
            ]
        )
        output = output.transpose(1, 2).reshape(-1, self.H)[batch.flat_idx.to(self.device)]
        return hidden + output @ weights["self_attn.o_proj.weight"].T

    def moe(self, hidden, weights, mask=None, log=None):
        x = self.rms(hidden, weights["post_attention_layernorm.weight"])
        logits = x @ weights["mlp.gate.weight"].T
        if mask:
            logits[:, list(mask)] = -torch.inf
        probabilities = logits.softmax(dim=-1)
        top_weights, top_indices = probabilities.topk(self.K, dim=-1)
        if self.cfg.norm_topk_prob:
            top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        if log is not None:
            log.append(
                (
                    to_numpy(top_indices).astype(np.int32),
                    to_numpy(top_weights).astype(np.float16),
                    to_numpy(probabilities).astype(np.float16),
                )
            )
        output = torch.zeros_like(hidden)
        for expert in range(self.E):
            rows, columns = torch.where(top_indices == expert)
            if len(rows):
                selected = x[rows]
                activated = F.silu(selected @ weights["G"][expert].T)
                activated = activated * (selected @ weights["U"][expert].T)
                value = activated @ weights["D"][expert].T
                output.index_add_(0, rows, value * top_weights[rows, columns, None])
        return hidden + output

    def token_losses(self, hidden, batch, norm, head):
        normalized = self.rms(hidden, norm)
        targets = torch.full((batch.N,), -100, dtype=torch.long, device=self.device)
        positions = np.where(np.r_[batch.doc[1:] == batch.doc[:-1], False])[0]
        targets[torch.tensor(positions, device=self.device)] = batch.tok[positions + 1].to(
            self.device
        )
        losses = torch.zeros(batch.N, device=self.device, dtype=hidden.dtype)
        for start in range(0, batch.N, 2048):
            losses[start : start + 2048] = F.cross_entropy(
                normalized[start : start + 2048] @ head.T,
                targets[start : start + 2048],
                reduction="none",
                ignore_index=-100,
            )
        return to_numpy(losses), to_numpy(targets != -100)

    def _validate(self, batch, conditions, group):
        if group < 1:
            raise ValueError("group must be positive")
        if self.cfg.max_position_embeddings < batch.T:
            raise ValueError("max_tokens exceeds the model context window")
        if batch.tok.max() >= self.cfg.vocab_size:
            raise ValueError("Token ID exceeds the model vocabulary")
        for name, condition in conditions.items():
            if (
                name == "base"
                or not name
                or any(
                    c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                    for c in name
                )
            ):
                raise ValueError("Condition names must be safe filenames; 'base' is reserved")
            for layer, experts in condition.items():
                if not isinstance(layer, int) or not 0 <= layer < self.L:
                    raise ValueError(f"Invalid layer {layer!r}")
                if any(not isinstance(e, int) or not 0 <= e < self.E for e in experts):
                    raise ValueError("Invalid masked expert")
                if len(set(experts)) > self.E - self.K:
                    raise ValueError("Mask must leave at least top-k experts available")

    @torch.inference_mode()
    def run(self, batch, conditions, group=12, log_routing=False, verbose=True, on_layer=None):
        """Return per-token losses, valid-target mask, and optional baseline routing.

        Each condition maps a name to ``{layer_index: [masked_expert_ids]}``.
        The baseline is always included as ``base``. Last tokens in each
        document have zero loss and are excluded by the returned valid mask.
        """
        self._validate(batch, conditions, group)
        norm, head = self.get("model.norm.weight"), self.get("lm_head.weight")
        hidden = self._embed(batch)
        starts = sorted({min(c) for c in conditions.values() if c})
        caches, results = {}, {}
        routing = [] if log_routing else None
        for layer in range(self.L):
            if layer in starts:
                caches[layer] = hidden.clone()
            weights = self.load_layer(layer)
            attended = self.attn(hidden, weights, batch)
            if on_layer is not None:
                on_layer(
                    layer, self.rms(attended, weights["post_attention_layernorm.weight"]), weights
                )
            hidden = self.moe(attended, weights, log=routing)
            del attended
            del weights
        results["base"], valid = self.token_losses(hidden, batch, norm, head)
        del hidden
        for name, condition in conditions.items():
            if not condition:
                results[name] = results["base"].copy()
        for start in starts:
            names = [name for name, c in conditions.items() if c and min(c) == start]
            for offset in range(0, len(names), group):
                selected = names[offset : offset + group]
                outputs = self._conditions(batch, conditions, selected, start, caches[start])
                results.update(
                    {
                        n: self.token_losses(h, batch, norm, head)[0]
                        for n, h in zip(selected, outputs, strict=True)
                    }
                )
                del outputs
            del caches[start]
        if verbose:
            print(
                f"Completed {len(results)} conditions; baseline loss {results['base'][valid].mean():.4f}",
                flush=True,
            )
        return results, valid, routing

    def _conditions(self, batch, conditions, names, start, initial):
        hidden = [initial.clone() for _ in names]
        for layer in range(start, self.L):
            weights = self.load_layer(layer)
            for index, name in enumerate(names):
                hidden[index] = self.moe(
                    self.attn(hidden[index], weights, batch),
                    weights,
                    mask=conditions[name].get(layer),
                )
            del weights
        return hidden

    @torch.inference_mode()
    def run_ckpt(self, batch, conditions, outdir, group=10, on_done=None, verbose=True):
        """Persist atomic per-condition losses and resume only matching inputs.

        Model/config, ordered tokenized corpus, engine source, and each named
        mask are checked on restart. New conditions may be added to a run.
        Callbacks run after files are saved; they must be idempotent.
        """
        self._validate(batch, conditions, group)
        identity = hashlib.sha256(
            json.dumps(
                {
                    "model": self.store.fingerprint(),
                    "tokens": batch.ids,
                    "max_tokens": batch.T,
                    "implementation": {
                        name: hashlib.sha256(
                            Path(__file__).with_name(name).read_bytes()
                        ).hexdigest()
                        for name in ("engine.py", "model.py", "checkpoints.py", "devices.py")
                    },
                    "dependencies": {
                        name: version(name)
                        for name in ("torch", "transformers", "numpy", "safetensors")
                    },
                    "runtime": {
                        "device": self.selection.identity(),
                        "threads": torch.get_num_threads(),
                        "matmul_precision": torch.get_float32_matmul_precision(),
                    },
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        checkpoint = Checkpoints(outdir, identity, conditions)
        pending = {n: c for n, c in conditions.items() if not checkpoint.exists(f"losses/{n}.npy")}
        if not pending and checkpoint.exists("losses/base.npy") and checkpoint.exists("valid.npy"):
            return
        starts = sorted({min(c) for c in pending.values() if c}, reverse=True)
        needed = [s for s in starts if not checkpoint.exists(f"cache/h_in_L{s}.npy")]
        norm, head = self.get("model.norm.weight"), self.get("lm_head.weight")
        if needed or not checkpoint.exists("losses/base.npy") or not checkpoint.exists("valid.npy"):
            hidden = self._embed(batch)
            for layer in range(self.L):
                if layer in needed:
                    checkpoint.save(f"cache/h_in_L{layer}.npy", to_numpy(hidden))
                weights = self.load_layer(layer)
                hidden = self.moe(self.attn(hidden, weights, batch), weights)
                del weights
            losses, valid = self.token_losses(hidden, batch, norm, head)
            checkpoint.save("valid.npy", valid)
            checkpoint.save("losses/base.npy", losses)
            del hidden
            if on_done:
                on_done(["base"])
        for name, condition in pending.items():
            if not condition:
                checkpoint.save(f"losses/{name}.npy", checkpoint.load("losses/base.npy"))
                if on_done:
                    on_done([name])
        for start in starts:
            names = [n for n, c in pending.items() if c and min(c) == start]
            for offset in range(0, len(names), group):
                selected = names[offset : offset + group]
                initial = torch.from_numpy(checkpoint.load(f"cache/h_in_L{start}.npy")).to(
                    self.device
                )
                outputs = self._conditions(batch, conditions, selected, start, initial)
                for name, hidden in zip(selected, outputs, strict=True):
                    checkpoint.save(
                        f"losses/{name}.npy", self.token_losses(hidden, batch, norm, head)[0]
                    )
                del outputs, hidden, initial
                if on_done:
                    on_done(selected)
                if verbose:
                    print(
                        f"Layer {start}: saved {offset + len(selected)}/{len(names)} conditions",
                        flush=True,
                    )
