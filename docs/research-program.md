# Local MoE research program

This project is a reusable, locally executable reference library for measuring
how mixture-of-experts models route information, how expert functions differ,
and what changes when those functions are intervened on. Its research outputs
are inspectable evidence, concise reader summaries, and detailed analyses with
reproducible methods. The original CPU VPS was an incidental constraint.
Local execution with access to model internals is a project requirement.

## Research questions and execution contract

The central question is whether an apparent domain preference in routing
predicts a reproducible, domain-selective causal contribution to model behavior.
Related questions concern functional redundancy, changes during training,
and how results vary with context, language, architecture, and precision.

Experiments operate on downloaded local weights with access to router logits,
selected experts, gate weights, activations, expert outputs, losses, and
intervention points. Managed generation APIs cannot supply that contract.
Model and dataset acquisition are explicit network operations; research
computation and reporting run locally. CUDA, ROCm, MPS, and CPU remain supported
through the existing capability-tested selection policy.

The verified local host is an Apple M5 Max with 128 GiB unified memory. Model
selection should follow the scientific comparison and measured local resource
requirements. Active parameter counts describe computation, while all stored
parameters, live activations, geometry controls, and transfer buffers affect
memory and storage. Synthetic MPS speedups do not predict full-model runtime.

## Audit of the current project

| Area | Existing evidence or implementation | Consequence for further research |
| --- | --- | --- |
| Historical pilot | Five domains; 500 analysis, 200 held-out, and 1,140 subtopic documents; archived tables and plots | Useful hypotheses and a fixed input population for a new measured run |
| Historical provenance | Original weights, raw routing/geometry arrays, and per-token ablation losses are absent; the notes identify no exact model commit | A newly pinned checkpoint can reconstruct the study, but cannot establish exact original-checkpoint replication |
| Inference | `Engine` explicitly supports OLMoE, full multi-head attention, and float32 tensor staging | Other model families require architecture work and numerical validation |
| Routing analysis | Domain divergence, expert usage, token categories, and same-token routing comparisons | Association and lexical diagnostics exist; matched semantic/source interventions do not |
| Geometry | Flattened weight cosine, functional output cosine, centered outputs, Gaussian and norm-matched random controls | Low cosine alone does not establish distinct knowledge or causal independence |
| Ablation | Experts are masked before softmax/top-k, allowing replacement experts and changed gate weights | This measures removal with rerouting; fixed-routing output removal is a different experiment |
| Controls | Global-mass and on-domain-mass matching, held-out evaluation, document bootstrap intervals | Control balance, selection independence, source clustering, and multiplicity need stronger confirmatory treatment |
| Study scale | Sample config uses 64 tokens, 20 permutations, and three control seeds | These are pilot settings, not a justified confirmatory sample size or null distribution |
| Local verification | Synthetic numerical tests and synchronized CPU/MPS benchmarks | Establish software behavior and small-model timing; full-model scientific findings require new runs |

The audit follows [pilot records](pilot-notes.md), the
[archive inventory](../research/pilot/README.md), [verification](verification.md),
and the current [engine](../src/moe_pilot/engine.py),
[probes](../src/moe_pilot/probes.py),
[domain analysis](../src/moe_pilot/workflows/domain_report.py),
[condition metadata](../src/moe_pilot/interventions.py), and
[configuration](../configurations/pilot.toml).

## Model priorities

| Priority | Local model comparison | Scientific purpose | Implementation status |
| --- | --- | --- | --- |
| 1 | OLMoE 0924 base with several checkpoints from its released training trajectory | Reconstruct the pilot and test when routing preferences and ablation effects emerge | Closest to current support; pin each available checkpoint and verify compatibility/parity |
| 1 | OLMoE 0125 base, SFT, DPO, and final Instruct | Measure how post-training changes routing, functional geometry, and intervention effects | Same model family; validate each checkpoint and record its lineage and input formatting |
| 2 | Granite 3.1 1B-A400M and 3B-A800M base models | Affordable architecture/scale comparisons and repeated study development | New Granite adapter, including grouped-query attention and model-specific scaling |
| 3 | Qwen3-30B-A3B-Base and a matched post-trained variant | Test whether findings extend to a larger 128-expert model | New adapter, including attention projection geometry, grouped-query attention, and gate normalization |
| 4 | DeepSeekMoE-16B base | Compare routed experts with explicitly shared experts | New adapter for shared experts and mixed layer structure; preserve model-specific terms |
| Later | Qwen3.5-35B-A3B | Compare hybrid attention/state architectures with routed and shared experts | Substantial new hybrid architecture support; a separate research extension |

The [OLMoE base card](https://huggingface.co/allenai/OLMoE-1B-7B-0924)
documents intermediate checkpoint branches. The
[0125 base](https://huggingface.co/allenai/OLMoE-1B-7B-0125),
[SFT](https://huggingface.co/allenai/OLMoE-1B-7B-0125-SFT),
[DPO](https://huggingface.co/allenai/OLMoE-1B-7B-0125-DPO), and
[Instruct](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct) cards
identify a useful post-training sequence. Differences along that sequence are
observational comparisons of training stages; they do not isolate a training
algorithm independently of its data and other settings.

IBM's [Granite architecture table](https://huggingface.co/ibm-granite/granite-3.1-3b-a800m-base)
lists approximately 1.3B/3.3B total parameters and 32/40 experts for the two
compact MoE models. [Qwen3's base card](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base)
lists 30.5B total parameters, 3.3B activated, and 128 experts with eight selected.
The [DeepSeekMoE paper](https://arxiv.org/abs/2401.06066) describes isolating
shared experts. [Qwen3.5's model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
specifies interleaved Gated DeltaNet and gated-attention layers, with eight routed
and one shared expert activated per MoE block. These are candidate study
targets, not models already executed by this package.

## Studies with the highest scientific value

**Routing preference and causal contribution.** Extend the current mass-matched
ablations with fixed-routing removal and controlled output substitution.
Measure whether a routing-based specialist prediction survives independent
sources, matched token composition, and both intervention semantics. The
[detailed protocol](studies/causal-specialization.md) defines this first study.

**Lexical, source, language, and contextual effects.** Use paired paraphrases,
translations, formatting changes, and identical token types in different
contexts. Balance source and task across domains. Expand the current token
identity analysis into controlled comparisons that can separate these factors.
Treat each language separately before aggregating a multilingual category.

**Training and post-training trajectories.** Track routing, expert output
geometry, and held-out intervention effects at early/middle/late checkpoints
and through base/SFT/DPO/final post-training. Track expert indices within a
verified lineage; compare functional patterns rather than raw expert IDs across
independently trained families.

**Functional geometry and redundancy.** Evaluate expert outputs on the same
inputs using centered similarity, representation similarity such as CKA, and
null controls that preserve more than shape or norm. Raw flattened weight
cosine changes under internal neuron permutations that preserve an expert's
function. Use functional metrics and intervention/rescue evidence when making
claims about distinct computation. Pairwise joint removals can measure
non-additive effects, with the remaining expert capacity kept valid.

**Context, generation, and precision.** Expand beyond the first 64 tokens and
distinguish prompt processing from generated continuation. Add instrumented
local generation and task scoring before making claims about reasoning or
answer quality. Treat reduced precision or quantization as an experimental
factor: measure routing margins, expert-set changes, loss, and task accuracy
against a float32 reference on identical raw text.

## Infrastructure needed for those studies

Extend the package around the current configuration, engine, observer, and
provenance contracts. A model adapter should expose architecture-correct local
execution and instrumentation; it must not conceal incompatible attention,
shared-expert, or normalization behavior. Each supported family needs tests
against its local Transformers reference and real-device checks where present.

Intervention metadata must identify the intervention kind, gate policy,
strength, affected layers/experts, donor/control selection, and seed. These
fields become part of artifact and checkpoint identity. Study matrices should
enumerate independent model/checkpoint/data/seed configurations and aggregate
only completed, validated runs. New behavior belongs in reusable package
modules with a documented CLI; no standalone experiment scripts are needed.

Add microbatching and streaming artifact/statistic aggregation before combining
longer contexts with large corpora. Preserve document boundaries, paired
observations, and exact token accounting when batching. Cache policy and
precision must be recorded for both scientific and performance comparisons.

## Reader summaries and detailed analyses

The project overview should state supported models, the research questions,
and a few measured findings linked to their study reports. New scientific
findings should enter that summary after a completed full-model study.

Each completed study should provide a short reader summary, detailed methods
and analysis, and an evidence inventory. The summary states the question,
effect size and uncertainty, tested population, and practical interpretation.
The detailed analysis includes paired examples, control balance, all specified
outcomes, null results, and sensitivity checks. The inventory links immutable
model/data identities, configuration, software/device identity, raw document
observations, figure-source tables, and reproducible commands.

The first deliverables should be an archived-input OLMoE reconstruction and the
causal-specialization study. Follow with the OLMoE training/post-training
comparison, then one compact independent family. This gives the model adapters
and subsequent larger-model work a stable scientific protocol.

## Reading that informs the experiments

- [OLMoE](https://arxiv.org/abs/2409.02060): open weights, data, logs, and
  checkpoints support reconstructing and extending the original experiment.
- [OpenMoE](https://arxiv.org/abs/2402.01739): reports token-identity routing and
  early formation of routing assignments; motivates lexical controls and
  training trajectories.
- [The Myth of Expert Specialization in MoEs](https://arxiv.org/abs/2604.09780):
  a 2026 preprint arguing that representation geometry explains routing
  similarity; motivates testing causal effects beyond routing association.
- [Understanding Multilingualism in MoE LLMs](https://arxiv.org/abs/2601.14050):
  studies language and depth effects; motivates language-specific controls and
  replication rather than treating multilingual text as one homogeneous domain.
- [Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414):
  develops CKA for representation comparison; informs functional similarity
  measurement alongside expert-output cosine.

These papers motivate questions and methods. They do not establish the results
of this project's proposed studies or guarantee that an extension is novel.
