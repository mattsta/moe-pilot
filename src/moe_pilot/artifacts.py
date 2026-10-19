"""Small shared artifact operations with deterministic UTF-8 JSON encoding."""

import json
import os
import tempfile
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".json")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_corpus(path, *, subtopics=False):
    records = read_json(path)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Corpus must be a nonempty array: {path}")
    for record in records:
        keys = ("domain", "text", "subtopic") if subtopics else ("domain", "text")
        if not isinstance(record, dict) or any(
            not isinstance(record.get(key), str) or not record[key].strip() for key in keys
        ):
            raise ValueError(f"Invalid corpus record in {path}")
    return records
