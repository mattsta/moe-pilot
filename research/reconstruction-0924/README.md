# OLMoE archived input reconstruction

The pinned local model `allenai/OLMoE-1B-7B-0924` at revision
`6d84c48581ece794365f2b8e9cfb043c68ade9c5` has completed routing, geometry, held-out
ablations, and subtopic analysis on the archived input population.

Baseline evaluation covers 500 documents and
29,522 valid targets across 5 domains.
Mean loss is **2.130555 nats/token**. The held-out ablation run evaluated
420 declared conditions in addition to baseline.

| domain | documents | valid_targets | mean_loss_nats_per_token |
| --- | --- | --- | --- |
| code | 100 | 6277 | 1.81323 |
| math | 100 | 6254 | 1.86404 |
| multilingual | 100 | 5597 | 2.44842 |
| prose | 100 | 6300 | 2.51714 |
| science | 100 | 5094 | 2.02143 |

The single-expert contrasts below compare specialist removal with the mean
on-domain-routing-mass control at selected layer 7. Positive differences
mean greater specialist loss impact on its nominated domain. Values are
nats/token; `relative_mass_gap` is a fraction. Intervals are
unadjusted paired document bootstraps; these are exploratory comparisons.

| domain | paired_difference | ci95_low | ci95_high | relative_mass_gap |
| --- | --- | --- | --- | --- |
| code | 0.00118505 | -0.0013687 | 0.00405392 | 0.221602 |
| math | 0.000742047 | -0.003433 | 0.00544073 | 0.578376 |
| multilingual | 0.132493 | 0.0961961 | 0.17143 | 0.927075 |
| prose | -0.000648999 | -0.00330995 | 0.00199414 | 0.449959 |
| science | 0.00759094 | 0.00223233 | 0.013344 | 0.62523 |

Mass gaps quantify imperfect matching. Routing preference, low parameter
cosine, and an unadjusted loss contrast do not establish a semantic specialist.
The original corpus also associates domains with different dataset sources.
The [detailed analysis](analysis.md) separates these observations and limits.

This is a reconstruction with newly pinned weights. The archived notes do not
identify the exact original model commit, and the new run does not recover that
missing identity. [Evidence inventory](evidence.json) records the exact inputs,
software, device, and generated artifacts.
