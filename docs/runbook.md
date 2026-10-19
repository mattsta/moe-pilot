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
profile. No automatic push or pull-request CI is configured. Run validation
locally when needed; publishing commits does not launch a hosted test matrix.

## Measure performance

Measure the automatically selected device without downloading a model or
running a scientific experiment. CPU is used only if accelerator selection
actually falls back to it, or if explicitly requested:

```sh
uv run --locked moe-pilot benchmark --output runs/benchmarks/medium.json
```

`moe_pilot.benchmark.benchmark_engine` is the reusable library API. The CLI
defaults to deterministic synthetic weights: two layers, eight experts, hidden
size 512, intermediate size 768, vocabulary 1,024, and top-two routing. The
`--profile tiny` option uses the numerical smoke model. Neither model represents
the full historical OLMoE checkpoint. Defaults are one document of 64 tokens,
one warmup, three repeats, two host threads, and baseline inference only.

To measure a local real checkpoint with the same synthetic token inputs:

```sh
uv run --locked moe-pilot benchmark --model models/olmoe-0924 --devices auto --documents 1 --max-tokens 64 --output runs/benchmarks/local-model.json
```

Use `cuda` or `rocm` in place of `mps` on those hosts and retain the appropriate
uv wheel profile. Explicit devices must pass their capability probes. If `auto`
resolves to CPU, duplicate CPU measurements are removed.

Use `--workloads routing_geometry ablations` only when those measurements are
needed. GPU work is synchronized before and after each timed call. Backends
run sequentially, releasing each model cache before the next backend. Losses
must agree with the first requested device's warmup within `2e-4`
absolute/relative tolerance. With one device this checks repeat consistency;
it is not an independent numerical reference. A CPU comparison runs only with
an explicit `--devices cpu ...` request and is not a recurring release check.

Timings include engine layer reads, weight/activation transfers, computation,
and returned host arrays. They exclude imports, model creation, tokenization,
device probes, and JSON writes. Warmups populate the process/filesystem cache;
these are not cold-start timings. No saved inference result or checkpoint is
reused. The JSON contains raw elapsed samples, median/min/max, effective
devices, model configuration/identity, environment, and source hashes.
Throughput counts input tokens per engine call, including ablation calls,
and is not autoregressive generation throughput.

Treat the machine as shared. Compare workload, batch, precision, threads,
and timing boundaries before interpreting speedup. The synthetic benchmark
excludes corpus construction, statistics, plots, and persisted checkpoint I/O;
it cannot establish full-pipeline or scientific-result improvements.
See [local measurements and limitations](performance.md).

## Shared-workstation resource policy

All engine workflows default to streaming weights and an 8 GiB job budget.
They preserve 16 GiB of available host memory on this Mac; on smaller
hosts the reserve is capped at a quarter of physical memory. The reusable
`ResourcePolicy` checks weights, staging, activations, and routing estimates
before inference and checks memory again between layers. Budget or pressure
failures stop the job; they do not trigger a CPU retry. MPS/CUDA allocator
limits apply to this process only. A per-user compute lock prevents cooperating
local jobs from running inference concurrently, even across output directories.
These conservative admission checks cannot
guarantee a total RSS ceiling under every framework allocation or other apps'
concurrent allocations.

The source-balanced research configuration uses a smaller **5 GiB process
budget and 8 GiB available-memory reserve**, two host threads, and streaming
weights. Available memory counts free pages and bounded reclaimable file cache,
not all inactive anonymous pages. Any macOS pressure warning stops inference.
An estimated admission failure can halve a document batch **before inference**;
actual pressure or a failed inference stops the run. Research collection refuses
CPU selection and does not silently turn accelerator failures into CPU jobs.

`execution.max_memory_gib` and `execution.reserve_memory_gib` configure the
policy. Benchmark equivalents are `--max-memory-gib` and
`--reserve-memory-gib`. Raising a budget requires an explicit configuration
decision. Defaults use two host threads and one simultaneous condition.
Resident caching is opt-in and is rejected when the estimate exceeds budget;
the full OLMoE float32 cache alone is approximately 25.4 GiB.

For release validation, run the offline tiny-fixture test suite and package
build. Do not repeat a full-model CPU comparison, download models, or start a
research sweep as part of a release check. Use the existing independently
validated reconstruction evidence for its recorded model/source identities.
Source changes still require software tests; old evidence does not validate a
changed full-model implementation.

The checked-in reconstruction configuration now uses conservative execution
settings. The completed reconstruction retains its original configuration in
`experiment.json` and its frozen runtime. Do not overwrite that completed run;
select a fresh `paths.output` before executing changed settings or source.

`--weight-cache resident` measures an explicit resident-weight policy; default
`stream` reloads layers on each pass. Choose `--workloads baseline ablations` to
omit expensive all-layer geometry when measuring inference itself. Every
measurement records the cache policy and cached parameter bytes. Warmups for
resident measurements exclude initial weight population; report cold loading
separately rather than interpreting these as startup timings.

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

Full-checkpoint parity is an explicit diagnostic, not a prerequisite to each
run or release. The original conversion comparison is already recorded in
[the reconstruction evidence](../research/reconstruction-0924/full-model-verification.json).
Only when validating a new model implementation, compare against Transformers
on the selected device with bounded inputs and a deliberately provisioned
memory budget:

```sh
uv run --locked moe-pilot --config configurations/reconstruction-0924.toml verify-model --documents 5 --max-tokens 64 --output runs/verification/olmoe-0924.json
```

The verification selects available domains from `corpus.json`, checks every
layer and valid next-token loss, and reports router-set agreement separately.
The reference runs on the selected device and is released before engine
inference. This requires memory for a full float32 reference plus staging;
the default budget rejects oversized references before loading them. It uses
only local model/tokenizer files.
Its reference-loading and streaming timings have different boundaries and must
not be interpreted as a performance speedup measurement.

For a frozen study corpus, add `--partition discovery` to verify on discovery
inputs. The configured `execution.weight_cache` applies to verification. A
full OLMoE 0924 resident check uses approximately 27.26 GB of cached float32
parameters, excluding embedding rows and live activations. Set
`execution.weight_cache = "stream"` when resident weights do not fit. The study
configuration also sets a document microbatch size; document-level observations
reject truncation that would silently discard their declared answer targets.

## Prepare corpora

For the source-balanced causal study, follow the
[data design and acquisition procedure](studies/data-design.md). It uses pinned
local parquet inputs with explicit origin filtering and grouped discovery,
calibration, evaluation, and formatting-diagnostic partitions. The study
commands require the optional `data` dependencies.

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

## Export validated research evidence

### Source-balanced research and the results interface

After the pinned sources and models are acquired and the cohort prepared
([data design](studies/data-design.md)), use the reusable study commands:

```sh
uv run --locked moe-pilot --config configurations/causal-study.toml study-observe --catalog configurations/models.toml --names olmoe-0924 --partition discovery
uv run --locked moe-pilot --config configurations/causal-study.toml study-observe --catalog configurations/models.toml --names olmoe-0924 --partition calibration
uv run --locked moe-pilot --config configurations/causal-study.toml study-select
uv run --locked moe-pilot --config configurations/causal-study.toml study-observe --catalog configurations/models.toml --names olmoe-0924 --partition evaluation --interventions
uv run --locked moe-pilot --config configurations/causal-study.toml study-observe --catalog configurations/models.toml --names olmoe-0125-base olmoe-0125-sft olmoe-0125-dpo olmoe-0125-instruct --partition evaluation
uv run --locked moe-pilot --config configurations/causal-study.toml study-observe --catalog configurations/models.toml --names olmoe-0125-sft olmoe-0125-instruct --partition evaluation --formatting paired-chat
uv run --locked moe-pilot --config configurations/causal-study.toml study-analyse
uv run --locked moe-pilot report
uv run --locked moe-pilot serve-report
```

Open **http://127.0.0.1:8765**. The checked-in `research/report/index.html`
also opens directly offline. Use domain, layer, and control filters for the
reconstruction and the metric selector for checkpoint comparisons. Every
chart has numeric tables or linked figure-source data; study figures export
PNG, SVG, and PDF. The paired-format panel contrasts native chat with plain
prompts on exactly aligned question tokens. Print/save PDF exports the page.
The generated `research/report/analysis.md` explains the outcomes and limits.

`study-analyse` reads only completed, hash-validated observations; it does not
run models. It regenerates tables, intervals, figures, and reader documents.
`report` builds the static interface; `serve-report` imports no model framework
and binds loopback only. The preview refreshes progress and validated study
tables every 30 seconds. Rebuild `report` after analysis to update offline
snapshots, source downloads, and figure exports. API refresh failures preserve
the last displayed results and are visible on the refresh button.

Collection runs checkpoints sequentially under the shared inference lock.
Set `OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1` and launch with `nice -n 10` on a busy Mac. Stop with a
normal interrupt and rerun the identical frozen command to resume. Saved
batches retain document order, hashes, timings, actual counts, and peak RSS.
`--max-batches 1` is an admission-sized checkpoint, not a completed study.
Changing the runtime, configuration, tokenization, or protocol requires a new
output directory. Use a frozen package as described below for long runs.

The study declares six primary interaction tests (three domains × removal and
rerouting), with source/domain-stratified paired bootstrap intervals and Holm
correction. Calibrated control matching is a measured prerequisite for a
semantic-specialization interpretation; failed balance is retained visibly.
Plain-prompt stage contrasts and the subsequently chosen native-chat follow-up
are observational/exploratory. They are not official benchmark scores.

### Archived-input reconstruction

After every reconstruction stage completes:

```sh
uv run --locked moe-pilot --config configurations/reconstruction-0924.toml publish-reconstruction --output research/reconstruction-0924
```

Publication takes the run's writer lock, verifies recorded input/source and
output identities, checks dependency consistency, and exports CSV/JSON/PNG
artifacts plus generated reader and detailed analyses. Large arrays remain in
the local run with their hashes retained. It validates the source identities
of a frozen runtime, rather than substituting the current package version.

For long experiments during continued development, build and install a frozen
package before running. An explicit local runtime preserves the package source
used by provenance validation:

```sh
uv build
uv --no-cache pip install --no-deps --python .venv/bin/python --target runs/runtime/reconstruction-v1 dist/moe_pilot-0.1.0-py3-none-any.whl
PYTHONPATH=runs/runtime/reconstruction-v1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 runs/runtime/reconstruction-v1/bin/moe-pilot --config configurations/reconstruction-0924.toml run
```

Use a fresh runtime directory for changed source and retain the original
runtime, model, inputs, dependency environment, and experiment outputs with
the run. Do not install over a runtime that an active or published run uses.
