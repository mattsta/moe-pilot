# Local research completion checklist

The active program follows the local execution contract in
[the research program](research-program.md). A phase is complete only when its
implementation, measured execution, validation, and evidence are recorded.
Software checks and proposed methods do not count as completed science.

| Phase | Deliverable and completion evidence | Status |
| --- | --- | --- |
| 1 | Immutable OLMoE 0924 snapshot, archived inputs, and full-depth CPU-reference/device parity | Complete; [evidence](../research/reconstruction-0924/full-model-verification.json) |
| 2 | Executed archived-input reconstruction with routing, functional geometry, held-out ablations, subtopics, and provenance | Running from frozen package ba75545 |
| 3 | Validated fixed-routing removal, explicit rescaling, strength/restoration, substitution, and intervention diagnostics | CPU/real-MPS software validation passed; full-model diagnostic study pending |
| 4 | Frozen discovery/calibration/evaluation splits, measured matching balance, paired contrasts, multiplicity, and lexical/source controls | Pending |
| 5 | Completed causal-specialization study, including null results, paired uncertainty, sensitivity analyses, and a reader summary | Pending |
| 6 | Executed OLMoE training-checkpoint and 0125 base/SFT/DPO/final comparisons with lineage and formatting recorded | Pending |
| 7 | Validated Granite architecture support and an executed independent-family comparison | Pending |
| 8 | Local Qwen3 and shared-expert model comparisons after adapter validation and resource measurement | Pending |
| 9 | Executed context, functional-similarity, generation, and precision sensitivity studies with appropriate reference checks | Pending |
| 10 | Source-backed overview findings, detailed study analyses, evidence inventories, clean focused commits, and reproducible bring-up | Pending |

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
  The current suite passes 146 tests; CUDA/ROCm hardware tests are skipped on
  this host. This establishes software semantics, not a causal study finding.
