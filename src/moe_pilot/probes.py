"""Read-only inference observers for expert weight and activation geometry."""

import numpy as np
import torch
import torch.nn.functional as F

from moe_pilot.devices import to_numpy


def pair_cosine(vectors):
    normalized = F.normalize(vectors, dim=1)
    return to_numpy(normalized @ normalized.T)


def activation_cosine(outputs):
    normalized = F.normalize(outputs, dim=-1)
    cosine = torch.einsum("eph,fph->efp", normalized, normalized)
    return to_numpy(cosine.mean(-1)), to_numpy(cosine.abs().mean(-1))


class OrthogonalityProbe:
    """Collect geometry on a fixed token sample using a private random generator."""

    def __init__(self, token_indices, seed):
        self.indices = torch.tensor(token_indices, dtype=torch.long)
        self.random = torch.Generator(device="cpu").manual_seed(seed)
        self.matrices = {}
        self.inputs = []
        self.extra = []

    def __call__(self, layer, inputs, weights):
        sample = inputs[self.indices.to(inputs.device)]
        self.inputs.append(to_numpy(sample).copy())
        experts = len(weights["G"])
        for key, projection in (("G", "gate"), ("U", "up"), ("D", "down")):
            vectors = torch.stack([w.flatten() for w in weights[key]])
            self.matrices[f"w_{layer}_{projection}_real"] = pair_cosine(vectors)
            del vectors
            random = torch.randn(
                experts, weights[key][0].numel(), generator=self.random, dtype=inputs.dtype
            ).to(inputs.device)
            self.matrices[f"w_{layer}_{projection}_random"] = pair_cosine(random)
            del random
        outputs = []
        random_outputs = []
        for expert in range(experts):
            gate, up, down = (weights[key][expert] for key in ("G", "U", "D"))
            outputs.append((F.silu(sample @ gate.T) * (sample @ up.T)) @ down.T)
            random_weights = [
                torch.randn(w.shape, generator=self.random, dtype=inputs.dtype).to(inputs.device)
                * (w.norm() / np.sqrt(w.numel()))
                for w in (gate, up, down)
            ]
            gate, up, down = random_weights
            random_outputs.append((F.silu(sample @ gate.T) * (sample @ up.T)) @ down.T)
        real = torch.stack(outputs)
        for kind, values in (("real", real), ("random", torch.stack(random_outputs))):
            signed, absolute = activation_cosine(values)
            self.matrices[f"a_{layer}_{kind}_signed"] = signed
            self.matrices[f"a_{layer}_{kind}_abs"] = absolute
        centered = real - real.mean(0, keepdim=True)
        upper = np.triu_indices(experts, 1)
        centered_cosine = activation_cosine(centered)[0][upper]
        mean_direction = F.normalize(real.mean(0), dim=-1)
        energy = real.pow(2).sum((0, 1))
        self.extra.append(
            {
                "layer": layer,
                "mean_cos_raw": activation_cosine(real)[0][upper].mean(),
                "mean_cos_centered": centered_cosine.mean(),
                "mean_abs_cos_centered": np.abs(centered_cosine).mean(),
                "max_cos_centered": centered_cosine.max(),
                "frac_energy_along_mean_expert_dir": (
                    (real * mean_direction[None]).sum(-1).pow(2)
                    / real.pow(2).sum(-1).clamp_min(1e-30)
                )
                .mean()
                .item(),
                "top1_hidden_dim_energy_share": (
                    energy.max() / energy.sum().clamp_min(1e-30)
                ).item(),
                "top1_hidden_dim": int(energy.argmax()),
            }
        )
