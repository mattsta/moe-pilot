# Causal study data design

The causal comparison uses English multiple-choice tasks with a common
question, labeled choices, and answer format. The initial local cohort contains
180 discovery, 120 calibration, and 180 final evaluation documents. Each split
is balanced across three domains and two dataset sources. The original
five-domain reconstruction remains a separate study.

## Sources and eligible population

| Dataset | Immutable revision | Computing | Mathematics | Science |
| --- | --- | ---: | ---: | ---: |
| cais/mmlu | c30699e8356da336a370243923dbaf21066bb9fe | 288 | 848 | 1,239 |
| TIGER-Lab/MMLU-Pro | b189ec765aa7ed75c8acfea42df31fdae71f97be | 94 | 503 | 1,923 |

Counts are eligible local records after subject selection, origin filtering,
and a 900-byte rendered-input limit, before duplicate grouping. The limit also
applies to the spaced formatting variant. The
[configuration](../../configurations/causal-study.toml) enumerates all subject
mappings and sampling settings.

[MMLU-Pro's construction description](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
identifies original MMLU questions among its sources. Entries whose `src`
matches the frozen case-insensitive `mmlu` pattern are excluded from the new
MMLU-Pro cohort. This removes 1,928 records in the selected subject categories
before the byte-limit filter. Remaining origins include STEM EZ, TheoremQA,
and SciBench. The source inventory retains raw and eligible origin counts.

The raw [MMLU](https://huggingface.co/datasets/cais/mmlu) file has 14,042 rows;
the pinned MMLU-Pro test file has 12,032. Dataset family is balanced within
each domain. Option counts and difficulty still differ between the datasets;
analysis must preserve dataset strata and show source-specific effects.
Filtering declared origins does not establish semantic independence or exclude
overlap with model training data.

## Grouping and independent partitions

Question identity uses normalized Unicode, case-folded word text. It excludes
answer choices, so augmenting distractors cannot separate otherwise identical
questions. Exact identities and question word-five-gram Jaccard similarity of
at least 0.8 are joined transitively. Groups with conflicting domain labels
are excluded and counted explicitly.

Sampling assigns entire content groups to one partition. Each source/domain
cell contributes 30 discovery, 20 calibration, and 30 evaluation observations.
Selection uses seed 20261019 and no model outcomes. Insufficient independent
groups cause failure rather than reducing a declared sample size. The initial
grouping found 4,799 groups from 4,895 eligible records, merging 96 records.
The grouping is a conservative textual leakage check, not a semantic-equivalence
oracle.

Discovery ranks experts. Calibration constructs controls, measures matching
quality, and estimates effect variability. Final evaluation remains separate
from those choices. Any revised cohort or primary analysis needs a new frozen
configuration and output directory before its evaluation is inspected.

## Paired formatting diagnostics

Every evaluation document has a standard and spaced rendering. The spaced
version changes only separators between the question, choices, and answer
label. Question text, choice text, order, and correct answer remain identical.
Both observations retain the same content group and stay in evaluation.
The resulting 360 observations are paired measurements of 180 documents.

Each observation retains source revision, origin, subject, answer index,
group identity, and the answer's character offset. Overall language-model loss
and answer-target loss are distinct endpoints. Task accuracy uses separately
validated, context-stable next-token answer-letter scoring over the displayed
options. No machine-generated paraphrases or
unreviewed translations are presented as validated semantic controls.

The executed protocol follow-up uses the checkpoint's native chat wrapper
rather than the prepared spaced variant. It was selected after the plain-prompt
stage comparison and is explicitly exploratory. Both members keep question,
choice, and response-label strings fixed. Tokenizer offsets and exact sequence
matching identify a common contiguous question-token span covering at least
90% of the plain prefix. Routing and span loss use only that shared span,
with identical deterministic probe positions inside each pair. Plain and
chat observations retain the original content group, source, and domain.

## Acquisition and validation

```sh
uv run --locked --extra data moe-pilot --config configurations/causal-study.toml download-study-data
uv run --locked --extra data moe-pilot --config configurations/causal-study.toml inspect-study-data --output runs/data-audit/causal-study.json
uv run --locked --extra data moe-pilot --config configurations/causal-study.toml prepare-study
```

The first command downloads exact parquet files and records repository,
revision, filename, and SHA-256. Inspection and preparation reject changed bytes
or mismatched provenance. Preparation records input identities and all output
hashes through `ExperimentRunner`. Corpus text and downloaded sources stay
outside version control; the published evidence inventory should retain their
identities and partition summaries.
