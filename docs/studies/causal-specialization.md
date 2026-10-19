# Causal specialization study protocol

This local experiment tests whether domain-related expert routing
predicts domain-selective loss effects after accounting for routing mass,
lexical composition, and source. Its primary target is a pinned OLMoE
checkpoint. The implementation supplies the intervention modes below,
checkpointed diagnostics, mass controls, and the
[source-balanced data preparation](data-design.md). The frozen plan is recorded
before the held-out intervention evaluation. The current implementation and
results are in the [interactive report](../../research/report/index.html) and
[detailed analysis](../../research/report/analysis.md).

The executed selection/calibration stage found **no admissible controls for
any of the three domains** under its frozen five-feature calipers. The retained
contrasts are explicitly exploratory; final evaluation does not turn those
failed matches into a causal-specialization finding.

## Hypotheses and observations

The primary hypothesis is that experts selected on discovery data produce
greater held-out on-domain loss increases than comparable control experts,
and that this excess is greater on-domain than off-domain. Evaluate this as an
interaction effect, rather than labeling an expert necessary from a positive
loss delta alone.

A second hypothesis is that the excess persists after matching or conditioning
on source, token frequency/type, language, and position. A third is that the
conclusion is robust to fixed-routing removal versus expert exclusion with
rerouting. Treat these as distinct tests with explicit endpoints.

The unit of observation is a document, or a shared-content group for paired
paraphrases/translations. Tokens within a document are not independent samples.
The target population is the specified local evaluation corpus and model
checkpoint; generalization to another population is a separate replication.

## Local execution and model validation

Pin the model repository to an immutable commit, acquire weights explicitly,
and keep inputs/model files separate from a fresh run output directory. Use
the existing local device-selection policy and record the effective device.
Validate full-depth baseline losses against the local Transformers reference
on a bounded shared batch before launching the larger study.

Start with float32 and the current OLMoE engine. Record tokenizer identity,
attention configuration, top-k, probability normalization, and intervention
semantics. Test intended batch/context sizes locally for memory and runtime;
the synthetic benchmark does not size this experiment. Generation endpoints
require a separately validated local generation implementation.

For archived inputs, describe the run as a reconstruction with the newly pinned
checkpoint. The archived notes do not identify the exact original model commit.

## Data and selection independence

Create disjoint discovery, control-calibration, and final evaluation groups.
Use discovery observations to rank candidate specialists and choose layers.
Use calibration observations to construct controls and check balance. Freeze
conditions and analysis choices before evaluating the final group.

Sample more than one source per domain where feasible, and include sources
that contribute multiple domains. Otherwise source and domain may be
inseparable. Deduplicate across all splits and exclude shared source documents
or paired-content groups from crossing split boundaries. Record attribution,
source revisions, sampling seeds, grouping, and exclusions.

Build a smaller paired diagnostic corpus with paraphrases, formatting variants,
and checked translations. Preserve a common content-group identifier and
document how meaning preservation was reviewed. Include token-type/context
comparisons; different tokenizers do not produce interchangeable token IDs.

Retain the original 64-token prefix condition for continuity, then add longer
windows with explicit target spans. Compare identical raw text spans across
models. Absolute per-token losses from different tokenizers are not directly
comparable; use within-model paired deltas, task outcomes, and correctly
accounted bits per byte when comparing likelihood across tokenizers.

## Intervention definitions

| Intervention                                  | Operation                                                                                          | Interpretation                                                                                                        |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Exclusion with rerouting                      | Mask selected expert logits before softmax/top-k                                                   | Effect of excluding those experts while allowing gate redistribution and replacements                                 |
| Fixed-routing output removal                  | Preserve router choices on incoming states at each target site; zero selected output contributions | Effect of removing contributions without replacement; removed output mass is part of the treatment                    |
| Fixed-routing removal with explicit rescaling | Remove contributions and rescale according to a declared rule                                      | Sensitivity to output magnitude; rescaling is a separate treatment                                                    |
| Controlled substitution or patch              | Replace a selected contribution with a declared donor/control contribution                         | Tests whether the specific contribution matters beyond comparable activity; donor validity needs checks               |
| Strength sweep and restoration                | Vary intervention strength, then restore the original contribution                                 | Checks response shape and implementation consistency; exact restoration alone does not establish semantic specificity |

For OLMoE configurations with unnormalized selected gate weights, pre-softmax
masking can change both selected experts and gate scale. Save the relevant
before/after gate masses, replacement identities, and contribution norms so
those effects remain visible.

Fixed-routing modes freeze routing locally at each targeted operation. Hidden
state changes can alter downstream routing; the modes do not globally freeze a
baseline route trajectory. Strength sweeps blend original/transformed outputs
for rerouting and substitution, and attenuate targeted contributions for
removal/rescaling. Gate diagnostics for output blends describe the transformed
branch. Complete removal under rescaling has explicitly defined zero output.

Use sham interventions and exact restoration tests. Preserve the original
batch/document mapping in every condition and validate each mode against a
small local reference. Intervention kind and gate policy must invalidate
incompatible checkpoint reuse.

## Controls and matching quality

Retain the current disjoint control families matched on global routing mass
and on-domain routing mass. Add balance checks for selected frequency,
contribution norm, layer, number of masked experts, and the routing changes
caused by each intervention. Specify calipers and sufficient overlap before
examining the final outcome.

Report achieved differences, not just a label saying matched. When acceptable
controls cannot be constructed, report that comparison as unsupported by the
available overlap. Random control draws from one expert population are not
independent model-training replications.

Use enough control draws to assess matching variability and null stability;
choose the count from calibration and runtime measurements. The pilot's three
control seeds are a starting point for calibration, not a precision guarantee.

## Outcomes and uncertainty

For document j, define Δ(j, S) as its intervention loss minus baseline loss for
specialist set S. The on-domain specialist excess is the paired mean of
Δ(j, S) minus the mean corresponding control delta. Compare that excess with
the same statistic off-domain to estimate selectivity beyond the control.
Predeclare the weighting of off-domain groups and whether the loss mean is
document-weighted or token-weighted.

Report paired document effects, confidence intervals, on/off-domain contrasts,
and control balance. Resample paired-content groups together. Use source-aware
resampling when the sampling structure supports it and report the effective
number of sources. A confidence interval over documents within one trained
checkpoint does not establish variability across training runs.

Choose sample counts from a calibration estimate of effect variability and a
declared smallest effect of interest. Freeze a small primary endpoint family
and a multiplicity rule; separate exploratory expert/layer searches. Increase
permutation counts according to the required resolution. With the common
plus-one empirical tail estimator, 20 permutations permit a minimum p-value
of only 1/21; the current report uses descriptive null summaries rather than
that inferential estimator.

Report negligible and negative effects. Avoid declaring experts redundant
because a test did not detect an effect, or necessary because one removal
increased loss. Joint removals can estimate non-additivity using
loss(A∪B) − loss(A) − loss(B) + loss(baseline); compute paired uncertainty
and keep enough experts available for the model's top-k.

## Analysis and release artifacts

The reader summary should answer which effects survived controls, where they
failed to generalize, and the size and uncertainty of the differences. Include
a domain-by-intervention effect plot, control-balance table, and a small set of
paired examples selected by a declared rule.

The detailed analysis should show discovery and evaluation populations,
checkpoint identity, intervention definitions, gate/contribution changes,
source and lexical sensitivity, all primary outcomes, exploratory outcomes,
and null findings. Keep routing association, functional similarity, and
intervention effects distinct in the interpretation.

Retain configuration, immutable input identities, implementation/device
provenance, condition metadata, document-level paired observations, valid-token
counts, and figure-source tables with the local run. Produce summaries from
validated completed artifacts. Keep large arrays and model downloads outside
version control, following the existing runbook and contribution rules.

The experiment is ready for a causal-specialization summary when baseline/intervention
checks pass, discovery and evaluation are separated, controls meet declared
balance criteria, and every primary outcome and sensitivity analysis has a
traceable result. A null result with that evidence is a completed study.
A completed protocol can also report failed control overlap and exploratory
outcomes; that is a valid scientific result with a narrower interpretation,
not a successful confirmatory test of semantic specialization.
