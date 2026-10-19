# Architecture

`moe-pilot` has an installable library and an executable research pipeline. It
uses the standard [uv project layout](https://docs.astral.sh/uv/guides/projects/)
with exact dependency resolution in `uv.lock`. Python 3.12 is the development
default; package metadata permits 3.11–3.14. CI checks 3.11 and 3.12 on Linux and
3.12 on macOS. Other permitted interpreters are not yet CI-verified.

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| `model.py` | Local configuration, safetensors mapping, model identity |
| `engine.py` | Batches, attention, expert routing, losses, condition execution |
| `checkpoints.py` | Atomic arrays, resume manifest, condition identities |
| `corpora.py` | Seeded sampling, deduplication, exclusions, group-size checks |
| `controls.py` | Specialist ranking and routing-mass-matched control selection |
| `artifacts.py` | UTF-8 JSON artifact reads and writes |
| `metrics.py` | Entropy and Jensen–Shannon divergence in bits |
| `inventory.py` | Read-only inventory of inputs and historical tables |
| `smoke.py` | Offline numerical and checkpoint verification infrastructure |
| `workflows/` | Explicit research steps retaining the original artifact schemas |
| `cli.py` | Arguments, CPU thread policy, workspace selection, workflow dispatch |

Imports never download datasets, load a checkpoint, or execute an experiment.
`Engine` loads local configuration when constructed; tokenizer loading is lazy.
Only `download-model` and the three corpus builders intentionally access the
network. Each engine owns its state; multiple engines can coexist.

## Library use

Place reusable application code in a module. For example:

```python
from moe_pilot.engine import Engine


def evaluate_documents(model_directory, documents):
    engine = Engine(model_directory)
    token_ids = [engine.tok(text).input_ids for text in documents]
    batch = engine.batch(token_ids, max_tokens=64)
    losses, valid, routing = engine.run(batch, {"mask_expert_21": {0: [21]}}, log_routing=True)
    return {name: float(values[valid].mean()) for name, values in losses.items()}
```

`run` returns a condition-to-loss mapping, a valid-target Boolean mask, and an
optional baseline routing list. Each routing entry contains top-k indices,
top-k weights, and full probabilities. Loss vectors are flattened in document
order. The final token of each document has zero loss and `valid=False`.
Always apply the valid mask before averaging. Loss is in nats per token;
reporting entropy and JSD are in bits.

Unused experts have undefined conditional domain/subtopic probabilities and
specialization (`NaN` in CSVs). Aggregate statistics exclude those undefined
values; lift heatmaps render them neutrally. This avoids divide-by-zero warnings
or assigning an arbitrary preferred domain to an expert with no observations.

`run_ckpt(batch, conditions, directory)` persists the same losses. It supports
adding conditions but refuses changes to an existing named mask or input batch.
An empty condition explicitly reproduces the baseline. `base` is reserved.
Model shards are identified by local path, size, and modification time; this
detects routine changes but is not a cryptographic weight-content guarantee.
Use immutable model revisions for published research.

## Boundaries and supported model configurations

The engine computes on CPU in float32, streams one decoder layer at a time,
and executes under `torch.inference_mode`. It supports the original per-expert
safetensors and newer Transformers packed expert tensors, sharded or unsharded.
It supports SiLU, standard multi-head attention, unscaled RoPE, optional QKV
clipping, and either top-k normalization setting. Unsupported grouped-query
attention, attention bias, scaled RoPE, and non-SiLU activations fail explicitly.

The engine is configurable; the pilot's ablation and subtopic reports retain
experiment-specific choices: 16 layers, 64 experts, fixed specialist IDs,
selected layers 0/7/15, and the original domain names. They are reference
workflows for this checkpoint, not a generic framework for arbitrary MoEs.
The forward and main domain-analysis steps also support smaller models for
offline verification. Layer streaming reduces resident weights, but retaining
many conditions' hidden states and baseline caches still consumes RAM/disk.

The CLI temporarily selects its workspace with `contextlib.chdir` and restores
the caller's directory. Workflow functions use relative research paths and
are intended for the CLI's single-threaded orchestration. Use `Engine`, corpus
sampling, and metric APIs for embedding in services or concurrent applications.
There is no server or background worker requirement.

## Entry points and preservation

All application code lives under `src/moe_pilot/`; the root contains project
metadata and documentation entry points. Run workflows through the installed
`moe-pilot` CLI. The former top-level script names are no longer entry points.
Library callers import `Engine` and `Batch` from `moe_pilot.engine` and construct
an engine explicitly. `build-conditions` only generates masks; `ablate` performs
inference. `smoke` runs the offline engine check; regression tests live in `tests/`.

`research/pilot/` preserves archived results, plots, sampled corpora, and source
attribution as a self-contained historical workspace. Inspect it with
`moe-pilot --workspace research/pilot inventory`. Create new experiments in
fresh workspaces so archived evidence remains unchanged.
The old environment dump is retained in `docs/requirements-original.txt` solely
as historical evidence; `pyproject.toml` and `uv.lock` define the new environment.

Verification code belongs in reusable package modules or checked-in tests.
Do not use inline Python commands or throwaway validation scripts.
