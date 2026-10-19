# OLMoE-1B-7B-0924 expert orthogonality / specialization pilot (CPU, 2026-10-18)

Hardware: 8 vCPU, 15 GiB RAM (~4.5 GiB available), no GPU, Python 3.13.5, torch 2.14.1+cpu, transformers 5.20.1.
Model: allenai/OLMoE-1B-7B-0924 (16 layers x 64 experts, top-8, norm_topk_prob=False), bf16 weights upcast to fp32.
The full model does not fit in RAM, so forward.py streams it one decoder layer at a time from safetensors.
Validated against HF OlmoeForCausalLM (2-layer truncation): max |diff| 1.3e-5, top-8 sets identical. Full-depth LM loss is 2.13 nats/token.
Corpus: 5 domains x 100 texts, first 64 tokens each, 30,022 tokens total (corpus_sources.json).
Pipeline: build_corpus.py -> forward.py 64 -> analyze.py -> act_extra.py. Outputs are in results/_.csv and plots/_.png.
Raw data: routing.npz (top-8 idx/weights + full router probs per token per layer), ortho_raw.npz (64x64 cosine matrices).
Baselines: Gaussian matrices of the same shape (weights), random-weight experts with matched norms on the same inputs (activations),
and a document-level domain-label permutation null (20 permutations) for the JSD and specialization index.

## Follow-up (2026-10-18): causal ablations + subtopics

- moe_engine.py: batched/checkpointed version of the layer-streaming forward. It reproduces forward.py (loss 2.1305, identical top-8).
- Ablations: heldout.json (40 texts/domain, disjoint from corpus.json, first 64 tokens). Run by ablate.py (condition build), ablate_run.py, ablate_ondomain_ctrl.py, then ablate_analyze.py.
  Outputs are in results/ablations: losses/_.npy (per token), delta_loss_long.csv, ablation_summary.csv, delta_loss_matrix__.csv, ablation_vs_ondomain_mass_controls.csv, conditions*.csv.
  Plots are in plots/ablations.
- Subtopics: build_subtopics.py writes subtopic_corpus.json (19 subtopics x 60 texts) and subtopics.py analyzes it. Outputs are in results/subtopics and plots/subtopics.
