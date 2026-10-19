# Operator runbook

Run commands from the checkout root after installing
[uv](https://docs.astral.sh/uv/getting-started/installation/). No GPU, model
download, authentication, or source-data download is required for local checks.

## Bring-up and software checks

```sh
uv sync --locked --all-extras
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
uv run --locked moe-pilot --workspace research/pilot inventory
uv run --locked moe-pilot --workspace /tmp/moe-pilot-smoke smoke
uv build
```

`uv sync` includes development tools by default. Dataset downloads are in the
`data` extra. For a runtime-only environment without corpus download support:

```sh
uv sync --locked --no-dev
uv run --locked --no-dev moe-pilot --help
```

The smoke command constructs a tiny random local OLMoE, compares next-token
losses to Hugging Face, evaluates a mask, and verifies checkpoint resume. It
writes `results/smoke.json` within the selected workspace. This verifies
software behavior; it does not reproduce the pilot's research findings.

Linux uses the PyTorch CPU wheel index to avoid pulling CUDA dependencies.
macOS uses PyPI's native wheel. The lockfile resolves both. The CLI defaults
to eight CPU threads; use `--threads 2` on a smaller machine. The library does
not set process-global thread limits.

## Full research run

Use a fresh workspace to preserve the supplied historical outputs. Select an
immutable model commit SHA from the model repository and keep it with the
experiment record. Replace `MODEL_COMMIT_SHA` below with that actual SHA.
Model download is explicit and can consume many GB. Do not run this pipeline
as a routine installation or CI check.

```sh
uv run --locked moe-pilot --workspace runs/pilot download-model --revision MODEL_COMMIT_SHA
uv run --locked moe-pilot --workspace runs/pilot build-corpus
uv run --locked moe-pilot --workspace runs/pilot build-heldout
uv run --locked moe-pilot --workspace runs/pilot forward 64
uv run --locked moe-pilot --workspace runs/pilot analyze
uv run --locked moe-pilot --workspace runs/pilot activation-extra
uv run --locked moe-pilot --workspace runs/pilot build-conditions
uv run --locked moe-pilot --workspace runs/pilot ablate
uv run --locked moe-pilot --workspace runs/pilot analyze-ablations
uv run --locked moe-pilot --workspace runs/pilot ondomain-controls
uv run --locked moe-pilot --workspace runs/pilot analyze-ablations
uv run --locked moe-pilot --workspace runs/pilot build-subtopics
uv run --locked moe-pilot --workspace runs/pilot subtopics
```

Global options (`--workspace`, `--model-dir`, `--threads`) precede the command.
`--model-dir` defaults to `models/olmoe` inside the workspace. An absolute path
can share one immutable model between workspaces. A shortened forward, such as
`forward 16 2`, is useful for debugging but cannot feed the fixed full-model
ablation and subtopic workflows.

To reuse the supplied sampled corpora, copy `corpus.json`, `heldout.json`,
`subtopic_corpus.json`, and their source metadata from `research/pilot/` into
the new workspace and skip their builders. Subtopic source metadata is stored
at `research/pilot/results/subtopics/corpus_sources.json`. New builds
deduplicate samples and reject short source pools;
they may differ from the historical samples. There is no silent fallback to a
different math dataset. Dataset repository revisions are not yet pinned by the
builders; archive the actual JSON files and source revisions used in a paper.

The first ablation analysis expands aliases into `conditions_expanded.csv`.
On-domain controls depend on that file, so the two analysis invocations in the
sequence are intentional. Condition generation does not perform inference.
Both control builders accept `--control-candidates` to set random subset search
effort (defaults: 4,000 overall-mass and 6,000 on-domain-mass candidates). Reduced
values support software checks and change matching quality; record this
parameter with any reported experiment.

## Inputs and outputs

| Step | Required inputs | Main outputs |
| --- | --- | --- |
| `forward` | local model/tokenizer, `corpus.json` | `routing.npz`, `ortho_raw.npz`, `probe_moe_inputs.npy` |
| `analyze` | those two NPZs, tokenizer | `results/*.csv`, `results/counts/`, `plots/*.png` |
| `activation-extra` | model, probe NPY | `results/activation_orthogonality_extra.csv` |
| `build-conditions` | routing NPZ, expert domain CSV | `results/ablations/conditions.{json,csv}` |
| `ablate` | local model, held-out corpus, conditions | checkpoint manifest, losses/cache NPYs, delta-loss CSV |
| `analyze-ablations` | routing, condition metadata, held-out losses | summaries, expanded aliases, ablation plots |
| `ondomain-controls` | routing, expert distribution, expanded aliases, held-out corpus | extra masks and checkpointed losses |
| `subtopics` | local model, subtopic corpus, original expert distribution | subtopic routing, CSVs, plots |

## Stop, resume, and recover

Interrupt inference normally; finished checkpoint arrays remain reusable. Rerun
the same `ablate` or `ondomain-controls` command. Baseline caches and finished
conditions are skipped. Loss arrays are written using temporary files and
atomic replacement. The delta-loss summary deduplicates condition/domain keys
and repairs rows after a completed restart.

Checkpoints are single-writer. Do not run two processes against the same
workspace. The manifest binds inference implementation source, numerical dependency versions, model identity,
ordered token IDs, token limit, and named masks. A different model, corpus,
engine implementation, or changed mask requires a new output directory.
Legacy checkpoint directories without manifests are rejected, not adopted.
Moving model shards also changes their identity. A new earlier-starting
condition may require another baseline pass to populate its hidden-state cache.

CSV/plot workflows are rerunnable and replace their outputs. They do not have
transactional multi-file publication. Subtopic routing is cached by filename;
use a fresh workspace if its model, corpus, or tokenization changes. Interrupted
non-checkpointed forward/subtopic runs must be restarted. Keep raw artifacts
and the model immutable when producing a report.

## Troubleshooting

- **Missing local model config:** download the checkpoint explicitly or pass
  the correct `--model-dir`. For installation verification, use `smoke`.
- **Missing routing/orthogonality arrays:** the supplied CSVs are archived
  outputs; the raw arrays were not supplied. Run `forward` first.
- **Missing dataset library:** use `uv sync --locked --all-extras`.
- **Source has too few texts:** inspect the named dataset and split; do not
  silently substitute a source or relax a research corpus definition.
- **Memory pressure:** use fewer threads, smaller debug corpora/token limits,
  or smaller `group` values in the engine API. Full ablations retain multiple
  hidden states; disk caches also require substantial capacity.
- **Checkpoint inputs changed:** use a fresh workspace. Do not copy a manifest
  onto old arrays to bypass identity checks.

## Research release record

Archive the code commit, `uv.lock`, interpreter and dependency versions, model
commit, dataset revisions/source metadata, exact corpus files and hashes,
commands, token limits, seeds, hardware, raw arrays, checkpoint manifest, and
final tables/plots. Report exclusions, permutation counts, bootstrap procedure,
and uncertainty. The pilot uses short texts, modest sample counts, and selected
specialists; avoid generalizing its findings beyond that experiment.
