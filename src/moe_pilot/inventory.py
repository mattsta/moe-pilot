"""Read-only experiment inspection without model loading or directory creation."""

import csv

from moe_pilot.artifacts import load_corpus, read_json


def inspect_experiment(config):
    paths = config.paths
    lines = [
        f"Data: {paths.data}",
        f"Model: {paths.model}",
        f"Output: {paths.output}",
        f"Model config: {'present' if (paths.model / 'config.json').is_file() else 'missing'}",
    ]
    for name in ("corpus.json", "heldout.json", "subtopic_corpus.json"):
        path = paths.data / name
        lines.append(f"{name}: {len(load_corpus(path)) if path.is_file() else 'missing'}")
    for path in (paths.routing, paths.orthogonality):
        lines.append(f"{path.name}: {'present' if path.is_file() else 'missing'}")
    manifest = paths.output / "experiment.json"
    if manifest.exists():
        for step, record in read_json(manifest).get("steps", {}).items():
            lines.append(f"Step {step}: {record['status']}")
    for path in sorted(paths.tables.rglob("*.csv")):
        with path.open(newline="") as stream:
            reader = csv.reader(stream)
            columns = next(reader, [])
            rows = sum(1 for _ in reader)
        lines.append(f"{path.relative_to(paths.output)}: {rows} rows, {len(columns)} columns")
    return "\n".join(lines)
