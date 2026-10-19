# Operator runbook

## Bring-up and offline checks

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run:

```sh
uv sync --locked --all-extras
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -W error
uv run --locked moe-pilot smoke --output /tmp/moe-pilot-smoke.json
uv build
```

No GPU, source-data download, model weights, or authentication is needed for
these checks. Smoke constructs a tiny deterministic local model, compares
baseline losses with Hugging Face, evaluates a mask, and checks checkpoint
resume. Its result is software verification, not research evidence.

Linux resolves CPU-only PyTorch wheels; macOS uses native PyPI wheels.
`uv sync --locked --no-dev` installs runtime dependencies. Source-data builders
additionally require `uv sync --locked --extra data`. No inline Python is needed.

## Define an experiment

Start with `configurations/pilot.toml`. Paths resolve relative to that file,
regardless of the invoking directory. It defines:

- Separate model, data, and output directories.
- Model repository and exact commit SHA for explicit downloads.
- Token limit, CLI CPU threads, and concurrent condition group size.
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
  use a larger host. Geometry and multiple conditions retain additional tensors.
- **Unpinned dataset/model revision:** fill the exact repository commit SHA
  before downloading; there is no mutable-revision fallback.

Archive code commit, `uv.lock`, config, `experiment.json`, checkpoint manifests,
model/dataset revisions, corpus/source files, hardware, raw arrays, final tables,
and figures. Loss is in nats/token; information metrics are bits. Ablation
bootstrap intervals are document-level and unadjusted for multiple comparisons.
Short sampled texts and selected interventions limit scientific generalization.
