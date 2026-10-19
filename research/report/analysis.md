# Source-balanced local OLMoE study

Open [the interactive results explorer](index.html) for charts, filters, numeric tables, CSV downloads, and printable results. Figure exports include PNG, SVG, and PDF. This document is generated from the same validated tables.

## Findings and scope

Completed plain-prompt evaluations: **5 checkpoints × 180 held-out questions**. Each checkpoint is pinned to an immutable revision. The six equally weighted source/domain cells contain MMLU and origin-filtered MMLU-Pro questions in computing, mathematics, and science.

| Checkpoint          | Forced-choice accuracy (fraction) | 95% low  | 95% high |
| ------------------- | --------------------------------- | -------- | -------- |
| olmoe-0125-base     | 0.233333                          | 0.172222 | 0.294444 |
| olmoe-0125-dpo      | 0.222222                          | 0.166667 | 0.277778 |
| olmoe-0125-instruct | 0.216667                          | 0.161111 | 0.272222 |
| olmoe-0125-sft      | 0.216667                          | 0.161111 | 0.277778 |
| olmoe-0924          | 0.250000                          | 0.194444 | 0.305556 |

Accuracy means selecting the highest-scoring next-token answer letter among the supplied choices under a plain prompt. These are zero-shot subset measurements, **not official MMLU/MMLU-Pro scores or a general reasoning evaluation**. Loss per token covers the supplied question, options, and answer; normalized choice loss is a separate outcome in nats/question.

No post-training checkpoint has a paired 95% accuracy interval wholly above zero relative to 0125 base in this plain-prompt cohort. The intervals are unadjusted; failure to detect improvement does not establish equivalence.

| Checkpoint vs 0125 base | Metric     | Difference | 95% low   | 95% high |
| ----------------------- | ---------- | ---------- | --------- | -------- |
| olmoe-0125-dpo          | accuracy   | -0.011111  | -0.077778 | 0.055556 |
| olmoe-0125-dpo          | choice_nll | 0.686709   | 0.446911  | 0.922632 |
| olmoe-0125-dpo          | loss       | 0.636519   | 0.607571  | 0.665564 |
| olmoe-0125-instruct     | accuracy   | -0.016667  | -0.083333 | 0.050000 |
| olmoe-0125-instruct     | choice_nll | 0.824387   | 0.575241  | 1.075238 |
| olmoe-0125-instruct     | loss       | 0.739766   | 0.706641  | 0.773529 |
| olmoe-0125-sft          | accuracy   | -0.016667  | -0.088889 | 0.055556 |
| olmoe-0125-sft          | choice_nll | 0.664685   | 0.443581  | 0.887004 |
| olmoe-0125-sft          | loss       | 0.303846   | 0.285062  | 0.322355 |

**0/3 domains passed every frozen control-balance threshold.** A failed match cannot support a semantic-specialization claim. The comparison remains useful as an explicitly exploratory intervention contrast. Selection used only discovery routing and calibration mass/usage/norm; final losses were not selection inputs.

| Domain      | Expert    | Control   | Relative on-mass gap | Relative off-mass gap | On-usage gap | Off-usage gap | Norm gap |
| ----------- | --------- | --------- | -------------------- | --------------------- | ------------ | ------------- | -------- |
| computing   | 6.000000  | 27.000000 | 0.421339             | 0.402835              | 0.277202     | 0.648206      | 0.378708 |
| mathematics | 37.000000 | 27.000000 | 0.438558             | 0.143060              | 0.099800     | 0.455907      | 0.024322 |
| science     | 63.000000 | 16.000000 | 0.871218             | 0.239978              | 0.851774     | 0.385025      | 0.310663 |

**Primary interaction results:** science / remove (+0.003378 nats/token, Holm p=0.0318); science / reroute (+0.003476 nats/token, Holm p=0.0405) clear the declared Holm threshold. This threshold result does not repair failed control matching or establish equivalence for undetected effects.

## Held-out intervention findings

The 0924 model completed all 33 frozen intervention conditions on 180 held-out questions, plus baseline. Primary endpoints are three domains × removal/rerouting, with Holm correction across six tests. Rescaling and donor substitution are secondary. Restoration must reproduce baseline losses exactly before any contrast is analysed.

| Domain      | Mode       | On−off specialist−control (nats/token) | 95% low   | 95% high | Primary Holm p |
| ----------- | ---------- | -------------------------------------- | --------- | -------- | -------------- |
| computing   | remove     | -0.000476                              | -0.001242 | 0.000270 | 0.932000       |
| computing   | reroute    | -0.000190                              | -0.001007 | 0.000649 | 1.000000       |
| computing   | rescale    | -0.000390                              | -0.001091 | 0.000301 | —              |
| computing   | substitute | -0.000538                              | -0.001388 | 0.000306 | —              |
| mathematics | remove     | 0.000875                               | -0.000345 | 0.002721 | 1.000000       |
| mathematics | reroute    | 0.000198                               | -0.001248 | 0.002127 | 1.000000       |
| mathematics | rescale    | 0.000874                               | -0.000696 | 0.003333 | —              |
| mathematics | substitute | 0.001069                               | -0.000240 | 0.002933 | —              |
| science     | remove     | 0.003378                               | 0.001092  | 0.005820 | 0.031800       |
| science     | reroute    | 0.003476                               | 0.000968  | 0.006208 | 0.040500       |
| science     | rescale    | 0.003846                               | 0.001213  | 0.006694 | —              |
| science     | substitute | 0.005998                               | 0.002659  | 0.009646 | —              |

The source/domain-stratified paired bootstrap uses 10,000 replicates. The two-sided studentized permutation test permutes paired effects within source (9,999 permutations; plus-one tail estimate). Its interpretation requires the label-exchangeability/null assumptions; domain labels were not randomized. Confidence intervals are unadjusted. Failed control overlap remains a causal limitation even if a statistical contrast is positive.

[Per-document contrasts](data/study/causal_paired_documents.csv), [source/domain effects](data/study/causal_by_source_domain.csv), and [half-strength and joint-removal diagnostics](data/study/intervention_diagnostics.csv) are available in full. Source/lexical sensitivity residualizes domain membership on source indicators and standardized recorded token-composition features; this is an exploratory conditional association and does not repair missing control overlap.

[Gate and output diagnostics](data/study/intervention_gate_diagnostics.csv) retain targeted-token counts, selected mass before/after treatment, expert-set changes, and perturbation norms. They are input-token means per batch, including document-final positions; they are not document-weighted loss outcomes.

The [correct-answer-label sensitivity](data/study/answer_label_effects.csv) uses already collected full-vocabulary correct-label losses. It is exploratory and separate from the frozen primary text-loss family, with its own Holm correction across twelve domain × mode tests. A correct-label loss is not task accuracy and does not score generated answers.

| Domain      | Mode       | Answer-label on−off excess (nats) | 95% low   | 95% high  | Exploratory Holm p |
| ----------- | ---------- | --------------------------------- | --------- | --------- | ------------------ |
| computing   | remove     | 0.000627                          | -0.010021 | 0.010772  | 1.000000           |
| computing   | reroute    | -0.006716                         | -0.024007 | 0.009791  | 1.000000           |
| computing   | rescale    | -0.002773                         | -0.014702 | 0.008751  | 1.000000           |
| computing   | substitute | -0.000281                         | -0.014880 | 0.015207  | 1.000000           |
| mathematics | remove     | -0.000949                         | -0.010791 | 0.008775  | 1.000000           |
| mathematics | reroute    | -0.010665                         | -0.026437 | 0.006268  | 1.000000           |
| mathematics | rescale    | -0.009065                         | -0.021033 | 0.002741  | 1.000000           |
| mathematics | substitute | 0.002583                          | -0.018081 | 0.021728  | 1.000000           |
| science     | remove     | -0.003663                         | -0.008746 | 0.000985  | 1.000000           |
| science     | reroute    | -0.004282                         | -0.011752 | 0.002357  | 1.000000           |
| science     | rescale    | -0.003939                         | -0.009846 | 0.001386  | 1.000000           |
| science     | substitute | -0.014306                         | -0.024543 | -0.005332 | 0.031200           |

## Routing and expert-output changes

| Checkpoint vs 0125 base | Mean per-question routing JSD (bits) | 95% low  | 95% high |
| ----------------------- | ------------------------------------ | -------- | -------- |
| olmoe-0125-dpo          | 0.009673                             | 0.009065 | 0.010349 |
| olmoe-0125-instruct     | 0.010147                             | 0.009553 | 0.010799 |
| olmoe-0125-sft          | 0.008760                             | 0.008201 | 0.009386 |

Routing distributions normalize selected gate mass within each question. The comparison uses identical token IDs across the 0125 stages. A routing shift alone does not demonstrate better reasoning, semantic partitioning, or causal necessity.

[Raw selected gate mass](data/study/selected_gate_mass.csv) reports the per-question sum of selected layer-7 probabilities before that normalization, with source/domain-stratified uncertainty. It keeps gate scale visible alongside distributional changes.

Expert-output magnitude uses a deterministic matched token position per question. Raw output norms are divided by the document's mean across experts before computing paired log ratios. Incoming hidden states differ across checkpoints. Expert indices are compared only within the 0125 lineage. This measures relative magnitude, not output direction, functional equivalence, or knowledge.

## Native-chat protocol follow-up

Native-chat accuracy differences: olmoe-0125-instruct: -5.56 percentage points (95% interval -11.67 to +0.56); olmoe-0125-sft: -5.00 percentage points (95% interval -11.11 to +1.11). These are protocol effects on forced-letter scoring, not generated-answer quality.

Aligned question-span loss changes (chat minus plain): olmoe-0125-instruct: +0.156245 nats/token [0.135638, 0.176664]; olmoe-0125-sft: +0.000327 nats/token [-0.012970, 0.013239]. A wrapper can change routing and question likelihood even when the measured question tokens are exactly identical.

| Checkpoint          | Chat−plain metric (routing JSD unsigned) | Estimate  | 95% low   | 95% high | Pairs      |
| ------------------- | ---------------------------------------- | --------- | --------- | -------- | ---------- |
| olmoe-0125-instruct | accuracy                                 | -0.055556 | -0.116667 | 0.005556 | 180.000000 |
| olmoe-0125-instruct | choice_nll                               | 0.081844  | -0.229330 | 0.402876 | 180.000000 |
| olmoe-0125-instruct | aligned_loss                             | 0.156245  | 0.135638  | 0.176664 | 180.000000 |
| olmoe-0125-instruct | routing_jsd_bits                         | 0.005045  | 0.004721  | 0.005377 | 180.000000 |
| olmoe-0125-sft      | accuracy                                 | -0.050000 | -0.111111 | 0.011111 | 180.000000 |
| olmoe-0125-sft      | choice_nll                               | 0.376658  | 0.097975  | 0.656118 | 180.000000 |
| olmoe-0125-sft      | aligned_loss                             | 0.000327  | -0.012970 | 0.013239 | 180.000000 |
| olmoe-0125-sft      | routing_jsd_bits                         | 0.005490  | 0.005145  | 0.005842 | 180.000000 |

The wrapper is the checkpoint's recorded native user/assistant template. Question/choice strings and response-label strings stay fixed. The longest exactly identical contiguous question-token span covers at least 90% of the plain question prefix; loss scores only transitions inside that span, and routing samples only its tokens. Whole-prompt token losses are not contrasted because wrapper lengths differ. Deterministic probes use the same aligned position within each pair.

This follow-up was chosen **after inspecting the plain-prompt stage results**. It is exploratory and within-checkpoint; it is not a causal training-algorithm comparison or a test of free-form generated answer quality. All intervals are paired across content groups and stratified by source and domain.

## Archived-input reconstruction: corrected post-hoc interaction

| Domain       | On−off excess (nats/token) | 95% low   | 95% high | Holm p   |
| ------------ | -------------------------- | --------- | -------- | -------- |
| code         | 0.001904                   | -0.000814 | 0.004704 | 0.558900 |
| math         | 0.001910                   | -0.002690 | 0.006766 | 0.927800 |
| multilingual | 0.133974                   | 0.098041  | 0.172199 | 0.000500 |
| prose        | -0.000423                  | -0.003165 | 0.002352 | 0.927800 |
| science      | 0.007716                   | 0.001987  | 0.013448 | 0.050800 |

Only multilingual clears Holm's 0.05 threshold with the studentized test in this saved reconstruction. Science has Holm p=0.0508. The source/domain confounding and 22–93% control-mass gaps remain. This reanalysis uses saved arrays and no new model inference; it is post hoc, not a confirmatory semantic-specialization result.

## Sampling, reproducibility, and limits

Discovery has 180 questions, calibration 120, evaluation 180; cells contribute 30/20/30 each. Questions are grouped across sources by normalized identity or word-five-gram Jaccard ≥0.8, with transitive groups kept within a single split. MMLU-derived origins are excluded from MMLU-Pro. The rendered plain input is at most 900 UTF-8 bytes. These checks limit textual overlap; they do not establish semantic independence or absence from pretraining. Source-specific difficulty and option counts still differ.

Selection ranks discovery on/off routing mass ratios at layer 7. It takes the first of the top eight candidates admitting a calibration match; otherwise it retains the top expert and marks failure. Control candidates have discovery specificity at or below its median and exclude the top eight. Calipers are ≤10% for on/off mass and usage and ≤20% for output norm. No held-out outcome was used to relax these rules.

Interventions are local to layer 7. Removal keeps the local routing choices and zeros the targeted contribution; rerouting excludes its router logit and permits replacements. Rescaling and donor substitution have separate declared semantics. Downstream states and routes can change. Half-strength curvature and joint-removal nonadditivity are descriptive diagnostics, not proof of redundant knowledge.

Runs use local float32 streaming execution on MPS, two host threads, a 5 GiB process budget, and current 8 GiB available-memory reserve (earlier phases used 16 GiB). OS pressure stops inference. Admission can halve a batch before inference; it never retries a failed model run on CPU. Peak process RSS includes neither every system allocation nor a guarantee about other applications. Execution timings are operational records, not controlled speedup measurements.

All intervals condition on one checkpoint per stage, this selected cohort, and the recorded protocol. The training-stage sequence is observational; algorithms and training data change together. No independent-family, training-trajectory, multilingual replication, generated-answer, or precision study is claimed by these outputs.

Reproduce with the [operator runbook](../../docs/runbook.md). The [analysis record](data/study/analysis_record.json), [figure inventory](figures/manifest.json), [frozen plan](data/plan.json), and copied per-phase manifests record hashes, revisions, numerical dependencies, and producer identities. Retain the local raw arrays, source parquet files, frozen runtimes, model snapshots, configuration, and uv.lock to rerun inference. The HTML and figure exports require no model loading.
