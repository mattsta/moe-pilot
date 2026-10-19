"""Read-only inventory of research inputs and archived CSV outputs."""

import csv
import json
from pathlib import Path


def inspect_workspace(workspace, model_dir=None):
    workspace = Path(workspace)
    model_dir = Path(model_dir) if model_dir else workspace / "models/olmoe"
    lines = [
        f"Workspace: {workspace}",
        f"Model config: {'present' if (model_dir / 'config.json').is_file() else 'missing'}",
    ]
    for name in ("corpus.json", "heldout.json", "subtopic_corpus.json"):
        path = workspace / name
        if path.is_file():
            records = json.loads(path.read_text())
            lines.append(f"{name}: {len(records)} documents")
        else:
            lines.append(f"{name}: missing")
    for name in ("routing.npz", "ortho_raw.npz", "probe_moe_inputs.npy"):
        lines.append(f"{name}: {'present' if (workspace / name).is_file() else 'missing'}")
    tables = sorted((workspace / "results").rglob("*.csv"))
    lines.append(f"Result tables: {len(tables)} (historical outputs; not revalidated by inventory)")
    for path in tables:
        with path.open(newline="") as stream:
            reader = csv.reader(stream)
            columns = next(reader, [])
            rows = sum(1 for _ in reader)
        lines.append(f"  {path.relative_to(workspace)}: {rows} rows, {len(columns)} columns")
    return "\n".join(lines)
