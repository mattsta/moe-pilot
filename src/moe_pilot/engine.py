"""Layer-streaming inference and routing ablations for OLMoE on PyTorch devices.

The engine owns its configuration and tensor store. Constructing a batch does
not tokenize or download data. All inference runs without autograd.
"""

import hashlib
import json
from dataclasses import dataclass, field
from importlib.metadata import version
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from moe_pilot.artifacts import write_json
from moe_pilot.checkpoints import Checkpoints
from moe_pilot.devices import DeviceSelection, resolve_device, to_numpy
from moe_pilot.model import ModelStore, rope_theta
from moe_pilot.perturbations import Intervention
from moe_pilot.resources import ResourcePolicy, bounded_inference


@dataclass
class Batch:
    """Host-side token/document metadata with explicit right-padding indices."""

    ids: list[list[int]]
    T: int
    limit: int = field(init=False)

    def __post_init__(self):
        if type(self.T) is not int or self.T < 2:
            raise ValueError("max_tokens must be at least 2 for next-token evaluation")
        self.limit = self.T
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
        self.T = int(self.lens.max())
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

    def __init__(self, model_dir, *, device="auto", weight_cache="stream", resources=None):
        if weight_cache not in ("stream", "resident"):
            raise ValueError("weight_cache must be stream or resident")
        self.store = ModelStore(model_dir)
        self._model_identity = self.store.fingerprint()
        self.weight_cache = weight_cache
        self._weights = {}
        self._tensor_sizes = None
        self.resources = resources or ResourcePolicy()
        self.resources.check()
        self.selection = device if isinstance(device, DeviceSelection) else resolve_device(device)
        self.device = self.selection.device
        self.resources.limit_device(self.device)
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
        if self.weight_cache == "stream":
            return self.store.get(name).to(self.device)
        if name not in self._weights:
            self._weights[name] = self.store.get(name).to(self.device)
        return self._weights[name]

    @property
    def resident_weight_bytes(self):
        return sum(tensor.numel() * tensor.element_size() for tensor in self._weights.values())

    def _embed(self, batch):
        # Select token rows while staging from disk; do not upload an entire
        # embedding table just to retain a small corpus's hidden states.
        return self.store.embedding_rows(batch.tok).to(self.device)

    def rms(self, x, weight):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.cfg.rms_norm_eps) * weight

    def load_layer(self, layer):
        self.resources.check()
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

    def _expert_sum(self, x, weights, indices, gates):
        output = torch.zeros_like(x)
        for expert in range(self.E):
            rows, columns = torch.where((indices == expert) & (gates != 0))
            if len(rows):
                selected = x[rows]
                activated = F.silu(selected @ weights["G"][expert].T)
                activated = activated * (selected @ weights["U"][expert].T)
                value = activated @ weights["D"][expert].T
                output.index_add_(0, rows, value * gates[rows, columns, None])
        return output

    def moe(self, hidden, weights, intervention=None, *, layer=None, log=None, on_trace=None):
        x = self.rms(hidden, weights["post_attention_layernorm.weight"])
        logits = x @ weights["mlp.gate.weight"].T
        probabilities = logits.softmax(dim=-1)
        top_weights, top_indices = probabilities.topk(self.K, dim=-1)
        if self.cfg.norm_topk_prob:
            top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        original_weights, original_indices = top_weights, top_indices
        targets = intervention.mask.get(layer, ()) if intervention else ()
        targeted = torch.zeros_like(top_indices, dtype=torch.bool)
        if targets:
            for expert in targets:
                targeted |= top_indices == expert
            if intervention.mode == "reroute":
                logits[:, list(targets)] = -torch.inf
                probabilities = logits.softmax(-1)
                top_weights, top_indices = probabilities.topk(self.K, dim=-1)
                if self.cfg.norm_topk_prob:
                    top_weights = top_weights / top_weights.sum(-1, keepdim=True)
            elif intervention.mode in ("remove", "rescale"):
                top_weights = top_weights * (1 - intervention.strength * targeted)
                if intervention.mode == "rescale":
                    retained = top_weights.sum(-1, keepdim=True)
                    scale = original_weights.sum(-1, keepdim=True) / retained.clamp_min(1e-30)
                    top_weights = top_weights * torch.where(retained > 0, scale, 0)
            else:
                top_indices = top_indices.clone()
                for target, donor in intervention.donors[layer].items():
                    top_indices[original_indices == target] = donor
        output = self._expert_sum(x, weights, top_indices, top_weights)
        original = None
        if targets and (
            on_trace is not None
            or (intervention.mode in ("reroute", "substitute") and intervention.strength < 1)
        ):
            original = self._expert_sum(x, weights, original_indices, original_weights)
            if intervention.mode in ("reroute", "substitute") and intervention.strength < 1:
                output = original + intervention.strength * (output - original)
        if targets and on_trace is not None:
            removed = (original_weights * targeted).sum(-1)
            changed = (top_indices.sort(-1).values != original_indices.sort(-1).values).any(-1)
            on_trace(
                {
                    "mode": intervention.mode,
                    "strength": float(intervention.strength),
                    "gate_diagnostics": "transformed branch before reroute/substitution output blending",
                    "tokens": len(x),
                    "targeted_tokens": int(targeted.any(-1).sum().item()),
                    "original_selected_mass_mean": float(original_weights.sum(-1).mean().item()),
                    "target_selected_mass_mean": float(removed.mean().item()),
                    "post_selected_mass_mean": float(top_weights.sum(-1).mean().item()),
                    "zero_post_mass_tokens": int((top_weights.sum(-1) == 0).sum().item()),
                    "fraction_changed_expert_sets": float(changed.float().mean().item()),
                    "original_output_norm_mean": float(original.norm(dim=-1).mean().item()),
                    "output_delta_norm_mean": float((output - original).norm(dim=-1).mean().item()),
                }
            )
        if log is not None:
            log.append(
                (
                    to_numpy(top_indices).astype(np.int32),
                    to_numpy(top_weights).astype(np.float32),
                    to_numpy(probabilities).astype(np.float32),
                )
            )
        return hidden + output

    def token_losses(self, hidden, batch, norm, head):
        normalized = self.rms(hidden, norm)
        targets = torch.full((batch.N,), -100, dtype=torch.long, device=self.device)
        positions = np.where(np.r_[batch.doc[1:] == batch.doc[:-1], False])[0]
        targets[torch.tensor(positions, device=self.device)] = batch.tok[positions + 1].to(
            self.device
        )
        losses = torch.zeros(batch.N, device=self.device, dtype=hidden.dtype)
        for start in range(0, batch.N, 256):
            losses[start : start + 256] = F.cross_entropy(
                normalized[start : start + 256] @ head.T,
                targets[start : start + 256],
                reduction="none",
                ignore_index=-100,
            )
        return to_numpy(losses), to_numpy(targets != -100)

    def _validate(self, batch, conditions, group):
        if self.store.fingerprint() != self._model_identity:
            raise ValueError("Model inputs changed after Engine construction; create a new engine")
        if group < 1:
            raise ValueError("group must be positive")
        if self.cfg.max_position_embeddings < batch.limit:
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
            if not isinstance(condition, Intervention):
                raise ValueError("Conditions must contain explicit Intervention objects")
            condition.validate(self.L, self.E, self.K)
        self.resources.check(self.estimate_memory(batch, conditions, group))

    def estimate_memory(self, batch, conditions, group):
        """Conservative weights, host staging, live activations, and routing allowance."""
        if self._tensor_sizes is None:
            self._tensor_sizes = {
                name: self.store.tensor_bytes(name) for name in self.store.weight_map
            }
        sizes = self._tensor_sizes
        if self.weight_cache == "resident":
            weights = sum(
                size for name, size in sizes.items() if name != "model.embed_tokens.weight"
            )
        else:
            weights = (
                max(
                    sum(
                        size
                        for name, size in sizes.items()
                        if name.startswith(f"model.layers.{layer}.")
                    )
                    for layer in range(self.L)
                )
                + sizes["lm_head.weight"]
                + sizes["model.norm.weight"]
            )
        starts = len({min(c.mask) for c in conditions.values() if c})
        states = batch.N * self.H * 4 * (16 + starts + min(group, len(conditions)))
        attention = len(batch.ids) * batch.T * self.H * 4 * 12
        logits = min(batch.N, 256) * self.cfg.vocab_size * 4 * 3
        routing = batch.N * self.L * (self.E + 2 * self.K) * 8
        return weights * 2 + states + attention + logits + routing

    @bounded_inference
    @torch.inference_mode()
    def run(
        self,
        batch,
        conditions,
        group=1,
        log_routing=False,
        verbose=True,
        on_layer=None,
        on_intervention=None,
    ):
        """Return per-token losses, valid-target mask, and optional baseline routing.

        Each condition maps a name to an explicit ``Intervention``.
        The baseline is always included as ``base``. Last tokens in each
        document have zero loss and are excluded by the returned valid mask.
        """
        self._validate(batch, conditions, group)
        norm, head = self.get("model.norm.weight"), self.get("lm_head.weight")
        hidden = self._embed(batch)
        starts = sorted({min(c.mask) for c in conditions.values() if c})
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
            names = [name for name, c in conditions.items() if c and min(c.mask) == start]
            for offset in range(0, len(names), group):
                selected = names[offset : offset + group]
                outputs = self._conditions(
                    batch, conditions, selected, start, caches[start], on_intervention
                )
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

    def _conditions(self, batch, conditions, names, start, initial, on_intervention=None):
        hidden = [initial.clone() for _ in names]
        for layer in range(start, self.L):
            weights = self.load_layer(layer)
            for index, name in enumerate(names):

                def trace(record, name=name, layer=layer):
                    on_intervention(name, layer, record)

                hidden[index] = self.moe(
                    self.attn(hidden[index], weights, batch),
                    weights,
                    intervention=conditions[name],
                    layer=layer,
                    on_trace=trace if on_intervention is not None else None,
                )
            del weights
        return hidden

    @bounded_inference
    @torch.inference_mode()
    def run_ckpt(
        self,
        batch,
        conditions,
        outdir,
        group=1,
        on_done=None,
        verbose=True,
        record_diagnostics=False,
    ):
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
                    "max_tokens": batch.limit,
                    "padding_width": batch.T,
                    "implementation": {
                        name: hashlib.sha256(
                            Path(__file__).with_name(name).read_bytes()
                        ).hexdigest()
                        for name in (
                            "engine.py",
                            "model.py",
                            "checkpoints.py",
                            "devices.py",
                            "perturbations.py",
                            "resources.py",
                        )
                    },
                    "dependencies": {
                        name: version(name)
                        for name in ("torch", "transformers", "numpy", "safetensors")
                    },
                    "runtime": {
                        "device": self.selection.identity(),
                        "threads": torch.get_num_threads(),
                        "matmul_precision": torch.get_float32_matmul_precision(),
                        "weight_cache": self.weight_cache,
                    },
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        checkpoint = Checkpoints(outdir, identity, {n: c.record() for n, c in conditions.items()})
        pending = {
            n: c
            for n, c in conditions.items()
            if (
                not checkpoint.exists(f"losses/{n}.npy")
                or (record_diagnostics and not checkpoint.exists(f"diagnostics/{n}.json"))
            )
        }
        if not pending and checkpoint.exists("losses/base.npy") and checkpoint.exists("valid.npy"):
            return
        starts = sorted({min(c.mask) for c in pending.values() if c}, reverse=True)
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
                if record_diagnostics:
                    write_json(checkpoint.path(f"diagnostics/{name}.json"), [])
                checkpoint.save(f"losses/{name}.npy", checkpoint.load("losses/base.npy"))
                if on_done:
                    on_done([name])
        for start in starts:
            names = [n for n, c in pending.items() if c and min(c.mask) == start]
            for offset in range(0, len(names), group):
                selected = names[offset : offset + group]
                initial = torch.from_numpy(checkpoint.load(f"cache/h_in_L{start}.npy")).to(
                    self.device
                )
                diagnostics = {name: [] for name in selected}

                def trace(name, layer, record, records=diagnostics):
                    records[name].append({"layer": layer, **record})

                outputs = self._conditions(
                    batch,
                    conditions,
                    selected,
                    start,
                    initial,
                    trace if record_diagnostics else None,
                )
                for name, hidden in zip(selected, outputs, strict=True):
                    if record_diagnostics:
                        write_json(checkpoint.path(f"diagnostics/{name}.json"), diagnostics[name])
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
