# Local model acquisition and comparison catalog

[The model catalog](../configurations/models.toml) records immutable checkpoint
revisions, local names, family, lineage, training stage, and analysis layers.
Model directories resolve relative to its declared root. Catalog targets
compose with the existing `Experiment` configuration for study execution.

## Acquire exact snapshots

```sh
uv run --locked moe-pilot download-models --catalog configurations/models.toml --names olmoe-0125-base olmoe-0125-sft olmoe-0125-dpo olmoe-0125-instruct
```

For the released training trajectory:

```sh
uv run --locked moe-pilot download-models --catalog configurations/models.toml --names olmoe-0924-step10000 olmoe-0924-step500000 olmoe-0924-step1000000
```

Acquisition is explicit and local execution uses the resulting files. Every
new snapshot records repository/revision, local file sizes/checksums, and any
conversion. A directory already recorded for another revision is rejected.
An intact existing snapshot is reused without changing its identity record,
which preserves the provenance of published experiments.

## Tensor format normalization

The OLMoE 0125 SFT release contains indexed PyTorch tensor `.bin` shards rather
than safetensors. The acquisition infrastructure loads them with
`torch.load(weights_only=True, map_location="cpu", mmap=True)`, validates tensor
keys against the source index, and writes safetensors with identical tensor
values and dtypes. It does not instantiate a serialized model object.

Source shards remain in place. Conversion records identify source and target
checksums, tensor counts, dtypes, and the conversion implementation hash. The
safe weight index points to the converted shards. The real SFT conversion
preserved BF16 tensors; inference continues to use declared float32 staging.
Converted and native checkpoints both require the same local numerical
validation before scientific comparison.

## Comparisons and interpretation

The catalog's three training points are approximately 41B, 2,097B, and 4,194B
training tokens, as labeled by the released OLMoE 0924 branches. The final
0924 base is a separate endpoint. The January 2025 base/SFT/DPO/final sequence
uses its own lineage. Expert IDs can be followed within a verified lineage;
they are not aligned semantic identities between different families.

The [OLMoE model card](https://huggingface.co/allenai/OLMoE-1B-7B-0924) documents
training checkpoint branches. The [January base](https://huggingface.co/allenai/OLMoE-1B-7B-0125)
and [final post-trained card](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct)
describe the later release and adaptation stages. Comparisons of these
checkpoints are observational training-stage comparisons; data and other
training differences are not isolated algorithmic treatments.
