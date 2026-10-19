# Operator runbook

## Bring-up and offline checks

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run:

```sh
uv sync --locked --extra data
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -W error
uv run --locked moe-pilot devices
uv run --locked moe-pilot smoke --output /tmp/moe-pilot-smoke.json
uv build
```

No GPU, source-data download, model weights, or authentication is needed for
these checks. Smoke constructs a tiny deterministic local model, compares
baseline losses with Hugging Face, evaluates a mask, and checks checkpoint
resume. Its result is software verification, not research evidence.

`uv sync --locked --no-dev` installs runtime dependencies. Source-data builders
additionally require `uv sync --locked --extra data`. No inline Python is needed.

## Devices and PyTorch installation

The experiment defaults to `execution.device = "auto"`: try CUDA or ROCm GPUs,
then MPS, then CPU. Each available candidate must pass synchronized float32
workload checks before being selected. `moe-pilot devices` shows the selected
backend and rejection reasons. CPU is reached only after accelerator candidates
are unavailable or fail capability checks.

PyTorch CUDA and ROCm are separate builds. The default locked install uses
PyPI's CUDA-capable Linux wheel and MPS-capable macOS wheel. On Linux, choose
the wheel profile matching your host:

```sh
# NVIDIA CUDA, or native Apple MPS on macOS:
uv sync --locked --extra data
uv run --locked moe-pilot devices

# AMD ROCm 7.2 on Linux:
uv sync --locked --extra data --extra rocm
uv run --locked --extra rocm moe-pilot devices
uv run --locked --extra rocm moe-pilot --config configurations/pilot.toml run

# Optional CPU-only Linux wheel profile:
uv sync --locked --extra data --extra cpu
uv run --locked --extra cpu moe-pilot smoke --device cpu
```

Retain the chosen `--extra rocm` or `--extra cpu` on `uv run` commands so uv
does not replace that variant with the default wheel. `cpu` and `rocm` are
mutually exclusive; `--all-extras` is intentionally unsupported. Source-data
commands additionally use `--extra data`. Runtime device selection cannot add
GPU support to a CPU-only wheel or install missing drivers. See
[uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) and
[PyTorch's platform installation guidance](https://docs.pytorch.org/get-started/locally/).

Set `execution.device` to `cuda`, `cuda:1`, `rocm`, `rocm:1`, `mps`, or `cpu` to
require that device. A forced device never silently falls back. For independent
hardware verification (no experiment config required):

```sh
uv run --locked moe-pilot smoke --device mps --output /tmp/moe-pilot-mps.json
uv run --locked moe-pilot smoke --device cuda --output /tmp/moe-pilot-cuda.json
uv run --locked --extra rocm moe-pilot smoke --device rocm --output /tmp/moe-pilot-rocm.json
```

`pytest -W error` exercises each real accelerator advertised by the host and
skips only absent hardware. Backend-order/failure tests run everywhere with
mocks. Inference stays float32; accelerator parity tests allow a documented
`2e-4` absolute/relative tolerance against the CPU reference.
The optional `Accelerator validation` workflow can be dispatched for CUDA,
ROCm, or MPS on a self-hosted runner labeled `gpu` and its Linux/macOS platform.
It first requires the requested device, then runs the suite with that wheel
profile. Normal hosted CI uses the optional CPU Linux wheel profile for speed;
macOS retains native MPS support and exercises it whenever the host exposes it.

## Measure performance

Compare CPU with the automatically selected accelerator without downloading a
model or running a scientific experiment:

```sh
uv run --locked moe-pilot benchmark --output runs/benchmarks/medium.json
```

`moe_pilot.benchmark.benchmark_engine` is the reusable library API. The CLI
defaults to deterministic synthetic weights: two layers, eight experts, hidden
size 512, intermediate size 768, vocabulary 1,024, and top-two routing. The
`--profile tiny` option uses the numerical smoke model. Neither model represents
the full historical OLMoE checkpoint. Default batches contain 1 and 16 documents
of 64 tokens, with two warmups, five timed repeats, and eight host threads.

To measure a local real checkpoint with the same synthetic token inputs:

```sh
uv run --locked moe-pilot benchmark --model models/olmoe --devices cpu mps --documents 1 16 --max-tokens 64 --warmup 2 --repeats 7 --threads 8 --output runs/benchmarks/local-model.json
```

Use `cuda` or `rocm` in place of `mps` on those hosts and retain the appropriate
uv wheel profile. Explicit devices must pass their capability probes. If `auto`
resolves to CPU, duplicate CPU measurements are removed.

Every case measures baseline inference, routing with all-layer geometry on up
to 100 sampled tokens, and baseline plus early/late single-expert ablations.
GPU work is synchronized before and after each timed call. Backend order
alternates between repeats. All timed losses must agree with CPU within
`2e-4` absolute/relative tolerance; numerical failures abort publication.

Timings include engine layer reads, weight/activation transfers, computation,
and returned host arrays. They exclude imports, model creation, tokenization,
device probes, and JSON writes. Warmups populate the process/filesystem cache;
these are not cold-start timings. No saved inference result or checkpoint is
reused. The JSON contains raw elapsed samples, median/min/max, effective
devices, model configuration/identity, environment, and source hashes.
Throughput counts input tokens per engine call, including ablation calls,
and is not autoregressive generation throughput.

Keep the machine otherwise idle. Compare workload, batch, precision, threads,
and timing boundaries before interpreting speedup. The synthetic benchmark
excludes corpus construction, statistics, plots, and persisted checkpoint I/O;
it cannot establish full-pipeline or scientific-result improvements.
See [local measurements and limitations](performance.md).

## Define an experiment

Start with `configurations/pilot.toml`. Paths resolve relative to that file,
regardless of the invoking directory. It defines:

- Separate model, data, and output directories.
- Model repository and exact commit SHA for explicit downloads.
- Compute device policy, token limit, CLI host threads, and condition group size.
- Random seed, permutation/bootstrap counts, probe sampling, and report layers.
- Expert counts and both control search efforts.
- Corpus group counts and dataset repository revisions.

Choose a new `paths.output` for different settings. Existing output directories
are bound to their recorded configuration, including download revisions.
Set the model's full 40-character commit SHA before downloading:

```sh
uv run --locked moe-pilot --config configurations/pilot.toml download-model
```

Downloads are explicit and can consume many GB. The command writes
`model_revision.json` alongside the local snapshot. An existing local model can
be used by setting `paths.model` directly; retain its identity with the run.

## Prepare corpora

For the supplied pilot inputs, copy the archived samples into the configured
data directory (the example config uses `data/pilot`):

```sh
mkdir -p data/pilot
cp research/pilot/corpus.json research/pilot/corpus_sources.json research/pilot/heldout.json research/pilot/heldout_sources.json research/pilot/subtopic_corpus.json data/pilot/
cp research/pilot/results/subtopics/corpus_sources.json data/pilot/subtopic_sources.json
```

The JSON texts are the inputs. Historical tables and plots stay in the archive
and are not loaded as active results. Preserving sampled JSON avoids silently
replacing the population of the historical pilot.

To build new corpora, specify exact repository commit SHAs in the config's
`[datasets]` table. Required repositories are listed in the example config.
Then run:

```sh
uv run --locked moe-pilot --config configurations/pilot.toml build-corpus
uv run --locked moe-pilot --config configurations/pilot.toml build-heldout
uv run --locked moe-pilot --config configurations/pilot.toml build-subtopics
```

Builders record source repository, revision, dataset configuration, split, and
fingerprint with group attribution. Samples are deduplicated, held-out texts
exclude the analysis corpus, and short source pools fail explicitly. Builders
never silently substitute datasets. Building samples uses seeds `analysis.seed`,
`analysis.seed + 1`, and `analysis.seed + 2` respectively.

## Execute and inspect

```sh
uv run --locked moe-pilot --config configurations/pilot.toml inventory
uv run --locked moe-pilot --config configurations/pilot.toml run
```

`inventory` is read-only. `run` executes these configured stages in order:

```sh
uv run --locked moe-pilot --config configurations/pilot.toml routing
uv run --locked moe-pilot --config configurations/pilot.toml domain-report
uv run --locked moe-pilot --config configurations/pilot.toml build-conditions
uv run --locked moe-pilot --config configurations/pilot.toml ablate
uv run --locked moe-pilot --config configurations/pilot.toml ablation-report
uv run --locked moe-pilot --config configurations/pilot.toml subtopic-report
```

The separate commands are equivalent to the ordered `run`. Global `--config`
precedes the command. Dataset construction and model downloading are separate
explicit operations; `run` does not fetch missing inputs.

| Stage              | Principal outputs within `paths.output`                                                                 |
| ------------------ | ------------------------------------------------------------------------------------------------------- |
| `routing`          | `artifacts/routing.npz`, `orthogonality.npz`, `probe_inputs.npy`; `tables/activation_orthogonality.csv` |
| `domain-report`    | `tables/*.csv`, `tables/counts/`, `figures/*.png`                                                       |
| `build-conditions` | `ablations/conditions.json` with specialists and both control families                                  |
| `ablate`           | `ablations/checkpoint/`, token/document metadata                                                        |
| `ablation-report`  | `tables/ablations/delta_loss.csv`, summary/matrix CSVs, `figures/ablations/delta_loss.png`              |
| `subtopic-report`  | `artifacts/subtopics.npz`, `tables/subtopics/`, `figures/subtopics/`                                    |

`experiment.json` records stage state, resolved configuration, implementation
and input identities, dependency versions, and output hashes. Reports require
valid upstream records and artifacts. Changing files by hand makes dependent
results stale; rerun the upstream stage instead of editing provenance records.

## Stop, resume, and recover

Interrupt normally and rerun `run` or the interrupted command. Completed stages
are skipped only after validating their dependencies and file hashes. Failed
and interrupted stages are never treated as completed output. A file lock
prevents concurrent writers to one experiment; the OS releases it after a crash.

The ablation engine additionally persists atomic per-condition loss arrays and
hidden-state caches. Completed conditions are skipped on restart. Model,
tokenized inputs, named masks, inference source, and numerical dependency
versions are checked. Changed settings/inputs require a new experiment output
directory; stale checkpoint files are not adopted. A crash after saving losses
can resume inference, then regenerate complete report tables deterministically.
The effective compute backend/device is part of both stage and checkpoint
identity, so a different automatic device selection cannot silently reuse
results produced on the previous backend.

Non-checkpointed routing and subtopic inference restart as a whole. CSV/figure
production is stage-scoped rather than a multi-file atomic transaction; a
failed stage's partial files cannot be consumed through the runner. Preserve
immutable inputs and use fresh output directories for changed experiments.

## Troubleshooting and release records

- **Missing input/model:** inspect the resolved paths with `inventory`; prepare
  the local inputs or use `smoke` for installation verification.
- **Missing/stale dependency:** run the named upstream stage, or use `run`.
- **Configuration changed:** select a new output directory.
- **Too many expert masks/control seeds:** masks must leave top-k experts, and
  the excluded pool must contain enough disjoint controls. Choose counts and
  seed counts appropriate to the configured model.
- **Memory pressure:** reduce condition groups, corpus/token/probe counts, or
  select a suitable device in a new experiment. Geometry and simultaneous
  conditions retain additional host/device tensors; a small startup capability
  probe cannot guarantee that a full experiment fits accelerator memory.
- **Unpinned dataset/model revision:** fill the exact repository commit SHA
  before downloading; there is no mutable-revision fallback.

Archive code commit, `uv.lock`, config, `experiment.json`, checkpoint manifests,
model/dataset revisions, corpus/source files, hardware, raw arrays, final tables,
and figures. Loss is in nats/token; information metrics are bits. Ablation
bootstrap intervals are document-level and unadjusted for multiple comparisons.
Short sampled texts and selected interventions limit scientific generalization.
