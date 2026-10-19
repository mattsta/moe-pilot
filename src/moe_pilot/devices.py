"""CUDA/ROCm, MPS, and CPU selection with workload capability checks."""

import platform
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


def to_numpy(tensor):
    """Artifact arrays always live on the host, independent of compute backend."""
    return tensor.detach().cpu().numpy()


def validate_device_name(requested):
    if not isinstance(requested, str) or not re.fullmatch(
        r"auto|cpu|mps|(?:cuda|rocm)(?::\d+)?", requested
    ):
        raise ValueError("device must be auto, cuda[:index], rocm[:index], mps, or cpu")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def probe_device(device):
    """Check float32 attention, RMS/router math, indexed accumulation, and loss.

    Compare representative operations with CPU and synchronize so asynchronous
    driver/kernel failures are rejected before a long experiment begins.
    """

    def evaluate(target):
        x = torch.arange(24, dtype=torch.float32, device=target).reshape(6, 4) / 24
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)
        attention = F.scaled_dot_product_attention(
            x[None, None], x[None, None], x[None, None], is_causal=True
        )
        logits = attention.reshape(6, 4)
        logits[:, 0] = -torch.inf
        probabilities = logits.softmax(-1)
        weight, index = probabilities.topk(2, dim=-1)
        rows, columns = torch.where(index == 3)
        output = torch.zeros_like(x)
        output.index_add_(0, rows, F.silu(x[rows]) * weight[rows, columns, None])
        output = output @ torch.eye(4, device=target, dtype=x.dtype)
        # Probe geometry kernels and the same ignore-index behavior as token loss.
        geometry = torch.einsum("eph,fph->efp", output[None], output[None])
        loss = F.cross_entropy(
            output,
            torch.tensor([1, 2, 3, 1, 2, -100], device=target),
            reduction="none",
            ignore_index=-100,
        )
        return torch.cat((loss, geometry.flatten()))

    expected = to_numpy(evaluate(torch.device("cpu")))
    actual = to_numpy(evaluate(device))
    synchronize(device)
    if not np.allclose(actual, expected, atol=2e-4, rtol=2e-4):
        raise RuntimeError("Required float32 operations disagree with CPU reference")


@dataclass(frozen=True)
class DeviceSelection:
    requested: str
    backend: str
    device: torch.device
    name: str
    rejected: tuple[str, ...] = ()

    def identity(self):
        return {
            "backend": self.backend,
            "device": str(self.device),
            "name": self.name,
            "machine": platform.machine(),
            "platform": platform.platform(),
            "torch_cuda": torch.version.cuda,
            "torch_hip": torch.version.hip,
            "dtype": "float32",
        }

    def record(self):
        return {"requested": self.requested, **self.identity(), "rejected": list(self.rejected)}


def _candidates():
    # CUDA and ROCm are mutually exclusive PyTorch builds. Both are exposed
    # through torch.cuda; HIP distinguishes AMD from NVIDIA at runtime.
    backend = "rocm" if torch.version.hip else "cuda"
    yield backend, "cuda", torch.cuda.is_available
    yield "mps", "mps", torch.backends.mps.is_available
    yield "cpu", "cpu", lambda: True


def resolve_device(requested="auto"):
    validate_device_name(requested)
    rejected = []
    for backend, tensor_backend, available in _candidates():
        family, _, index = requested.partition(":")
        if requested != "auto" and family != backend:
            continue
        try:
            if not available():
                rejected.append(f"{backend}: unavailable in this PyTorch build or host")
                continue
            indices = (
                [int(index)]
                if index
                else (
                    range(torch.cuda.device_count())
                    if tensor_backend == "cuda"
                    else ([0] if tensor_backend == "mps" else [None])
                )
            )
            for number in indices:
                device = (
                    torch.device(tensor_backend, number)
                    if number is not None
                    else torch.device(tensor_backend)
                )
                try:
                    probe_device(device)
                    name = (
                        torch.cuda.get_device_name(device)
                        if tensor_backend == "cuda"
                        else ("Apple Metal" if backend == "mps" else platform.processor() or "CPU")
                    )
                    return DeviceSelection(requested, backend, device, name, tuple(rejected))
                except (RuntimeError, NotImplementedError, AssertionError) as error:
                    rejected.append(f"{backend} {device}: {error}")
        except (RuntimeError, AssertionError) as error:
            rejected.append(f"{backend}: {error}")
    raise ValueError(f"No compliant device for {requested!r}: {'; '.join(rejected)}")
