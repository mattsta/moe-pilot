# Archived pilot evidence

This directory preserves the supplied pilot artifacts: 500 analysis documents,
200 held-out documents, 1,140 subtopic documents, source attribution, 37 CSV
tables, and plots. The supplied artifact organization is preserved here.
Original experiment notes and environment details are in
[the pilot notes](../../docs/pilot-notes.md) and
[the historical dependency dump](../../docs/requirements-original.txt).

This archive is historical evidence, separate from active experiment outputs.
Copy sampled inputs into an experiment's configured data directory to reuse
them; current workflows do not interpret archived tables as live results.

Model weights, raw routing/orthogonality arrays, and per-token ablation losses
were not supplied. These tables are historical evidence, not results newly
validated by the package's synthetic software checks. Dataset texts retain
their original source terms.

Use a new configured output directory for experiments. See [the operator runbook](../../docs/runbook.md)
for bring-up, corpus reuse, full pipeline execution, and checkpoint recovery.
