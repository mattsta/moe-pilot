# Archived pilot workspace

This directory preserves the supplied pilot artifacts: 500 analysis documents,
200 held-out documents, 1,140 subtopic documents, source attribution, 37 CSV
tables, and plots. The research file schema is unchanged within this workspace.
Original experiment notes and environment details are in
[the pilot notes](../../docs/pilot-notes.md) and
[the historical dependency dump](../../docs/requirements-original.txt).

From the repository root, inspect available artifacts with:

```sh
uv run --locked moe-pilot --workspace research/pilot inventory
```

Model weights, raw routing/orthogonality arrays, and per-token ablation losses
were not supplied. These tables are historical evidence, not results newly
validated by the package's synthetic software checks. Dataset texts retain
their original source terms.

Use a fresh workspace for new experiments. See [the operator runbook](../../docs/runbook.md)
for bring-up, corpus reuse, full pipeline execution, and checkpoint recovery.
