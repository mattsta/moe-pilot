# Local research completion checklist

The [interactive report](../research/report/index.html) is the reader entry point;
the [detailed analysis](../research/report/analysis.md) links all completed
comparisons, uncertainty, figures, and reproducibility records.

The active program follows the local execution contract in
[the research program](research-program.md). A phase is complete only when its
implementation, measured execution, validation, and evidence are recorded.
Software checks and proposed methods do not count as completed science.

The oversized resident-cache benchmark was stopped at the user's request. It
produced no accepted performance result. Recurring CPU comparisons have been
removed; ongoing inference defaults to streaming, a shared-host reserve, and
an explicit job budget. Release validation uses tiny offline fixtures. The
pending research phases below remain pending and are not release prerequisites.

| Phase | Deliverable and completion evidence | Status |
| --- | --- | --- |
| 1 | Immutable OLMoE 0924 snapshot, archived inputs, and full-depth CPU-reference/device parity | Complete; [evidence](../research/reconstruction-0924/full-model-verification.json) |
| 2 | Executed archived-input reconstruction with routing, functional geometry, held-out ablations, subtopics, and provenance | Complete; [reader summary](../research/reconstruction-0924/README.md), [analysis](../research/reconstruction-0924/analysis.md), [evidence](../research/reconstruction-0924/evidence.json) |
| 3 | Validated fixed-routing removal, explicit rescaling, strength/restoration, substitution, and intervention diagnostics | Complete for OLMoE 0924: all 33 conditions on 180 questions, exact restoration, half-strength response, joint removal, and gate/output traces. |
| 4 | Frozen discovery/calibration/evaluation splits, measured matching balance, paired contrasts, multiplicity, and lexical/source controls | Implemented and calibrated: 180/120/180 questions; all three strict control matches failed. Analysis reports that failure rather than relaxing the thresholds. |
| 5 | Completed causal-specialization study, including null results, paired uncertainty, sensitivity analyses, and a reader summary | Core protocol complete; control overlap failed, so outcomes remain exploratory. All six primary tests, secondary modes, source/lexical sensitivity, and answer-label diagnostics are published. |
| 6 | Executed OLMoE training-checkpoint and 0125 base/SFT/DPO/final comparisons with lineage and formatting recorded | All four 0125 plain-prompt evaluations complete (720 question/checkpoint observations); SFT and final native-chat follow-up complete (360 question pairs). Earlier training snapshots remain unexecuted. |
| 7 | Validated Granite architecture support and an executed independent-family comparison | Pending |
| 8 | Local Qwen3 and shared-expert model comparisons after adapter validation and resource measurement | Pending |
| 9 | Executed context, functional-similarity, generation, and precision sensitivity studies with appropriate reference checks | Paired native-chat context study complete on exactly aligned question tokens. Reconstruction geometry is complete; additional similarity, generation, and precision studies remain unexecuted. |
| 10 | Source-backed overview findings, detailed study analyses, evidence inventories, clean focused commits, and reproducible bring-up | Complete for the core studies: interactive report, PNG/SVG/PDF figures, generated detailed analysis, validated source tables/manifests, and documented bring-up. Future model-family studies remain separate. |

Hybrid Qwen3.5 support is a subsequent architecture extension. It is not claimed
as supported or completed by the studies above.

## Execution record

- Available disk space at start: approximately 841 GiB.
- OLMoE 0924 base resolved from the public model manifest to commit
  `6d84c48581ece794365f2b8e9cfb043c68ade9c5` on 2026-10-19.
- The manifest lists 6,919,161,856 BF16 parameters, approximately 13.84 GB of
  parameter data. Inference uses the current float32 staging policy.
- Original model commit identity is absent from the archive, so this run is a
  reconstruction using newly pinned weights.
- Full-depth verification: five documents, 286 input tokens, 281 valid targets;
  maximum absolute loss error `8.68e-5`; selected expert sets agree at every
  token in every layer. CPU mean loss `2.093628`, MPS mean loss `2.093667`.
- All four intervention modes and strengths 0, 0.5, and 1 were checked against
  independent Transformers expert execution. Real-MPS integration checks
  include removal, rescaling, substitution, restoration, and diagnostics.
  The final suite passes **208 tests**, including real-MPS checks; two
  CUDA/ROCm hardware tests are skipped because those backends are absent.
  This establishes software semantics, not a causal study finding.
- The [causal data design](studies/data-design.md) records source-origin
  exclusions, duplicate grouping, six balanced source/domain cells, and
  discovery/calibration/evaluation counts. No final model outcomes have been
  used to select this cohort.
- The reconstruction completed all six stages from frozen package ba75545:
  baseline loss 2.130555 nats/token over 29,522 valid targets, 420 held-out
  conditions, and the full subtopic analysis. Publication validates source,
  input, output, and dependency identities. The selected layer-7 single-expert
  comparisons show 22–93% relative control-mass gaps, motivating stricter
  matching rather than a claim of semantic expert necessity.
- A post-hoc studentized on/off interaction analysis of the saved reconstruction
  retains only multilingual after Holm correction across five domains;
  science has adjusted p=0.0508. Source/domain confounding and poor mass
  matching remain, so neither this nor the unadjusted intervals proves
  semantic specialization.
- All four 0125 stages completed the same 180 held-out questions on MPS.
  Plain-prompt forced-letter accuracy is 23.3% base, 21.7% SFT, 22.2% DPO,
  and 21.7% final. Paired accuracy intervals versus base include zero.
  Routing divergence and relative expert-output magnitude changes are in the
  report; they do not establish improved reasoning or distinct knowledge.
- Current research uses a 5 GiB process budget, an 8 GiB available-memory
  reserve, two threads, reduced scheduling priority, and one accelerator
  job at a time. Earlier discovery/calibration and base/SFT evaluations used
  a 16 GiB reserve. Exact policies and measured peak RSS remain per phase.
- The source-balanced 0924 evaluation completed 180 questions × 33 conditions
  plus baseline on MPS, from frozen runtime `causal-v3`, with measured peak
  process RSS **1.39 GiB**. Every zero-strength restoration loss equals baseline.
- Science on/off specialist-minus-control interactions are +0.003378 nats/token
  for removal (95% CI 0.001092–0.005820; Holm p=0.0318) and +0.003476 for
  rerouting (0.000968–0.006208; Holm p=0.0405). Computing and mathematics
  primary intervals include zero. All controls failed calibration balance;
  these results do not establish semantic specialization.
- Both native-chat comparisons completed 180 content pairs with 100% aligned
  question-token coverage. Accuracy differences were −5.00 pp for SFT
  (95% CI −11.11 to +1.11) and −5.56 pp for final (−11.67 to +0.56).
  Final-model aligned question loss increased by 0.1562 nats/token
  (0.1356–0.1767). This measures wrapper/context sensitivity, not generated-answer
  quality or a training-algorithm effect.
