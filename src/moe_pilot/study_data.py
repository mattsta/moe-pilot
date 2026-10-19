"""Pinned local study sources, origin audits, and grouped disjoint partitions."""

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from moe_pilot.artifacts import read_json, write_json


def _hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_record_path(source):
    return source.path.with_suffix(".source.json")


def acquire_sources(config):
    """Explicitly download exact dataset files and retain local byte identities."""
    from huggingface_hub import hf_hub_download

    if not config.study.sources:
        raise ValueError("Configure study.sources before acquiring study data")
    outputs = []
    for source in config.study.sources:
        parts = source.filename.split("/")
        if tuple(source.path.parts[-len(parts) :]) != tuple(parts):
            raise ValueError("Study source path must end with its repository filename")
        root = source.path.parents[len(parts) - 1]
        filename = hf_hub_download(
            source.repository,
            source.filename,
            repo_type="dataset",
            revision=source.revision,
            local_dir=root,
        )
        if source.path.resolve() != Path(filename).resolve():
            raise ValueError("Downloaded source path differs from configured input")
        metadata = {
            "repository": source.repository,
            "revision": source.revision,
            "filename": source.filename,
            "sha256": _hash(source.path),
        }
        write_json(source_record_path(source), metadata)
        outputs.extend((source.path, source_record_path(source)))
    return outputs


def render_question(question, choices, answer, *, spaced=False):
    separator = "\n\n" if spaced else "\n"
    prefix = separator.join(
        [
            f"Question: {question}",
            *[f"({chr(65 + index)}) {choice}" for index, choice in enumerate(choices)],
            "Answer:",
        ]
    )
    return prefix + f" {chr(65 + answer)}", len(prefix)


def _normalize_question(question):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", question).casefold()))


def _read_source(source, max_bytes):
    metadata = read_json(source_record_path(source))
    expected = {
        "repository": source.repository,
        "revision": source.revision,
        "filename": source.filename,
        "sha256": _hash(source.path),
    }
    if metadata != expected:
        raise ValueError("Study source bytes or declared immutable identity changed")
    table = pd.read_parquet(source.path)
    subject_key, choices_key, answer_key = (
        ("subject", "choices", "answer")
        if source.format == "mmlu"
        else ("category", "options", "answer_index")
    )
    required = {subject_key, choices_key, answer_key, "question"}
    if not required.issubset(table.columns):
        raise ValueError(
            f"Missing {source.format} columns: {sorted(required - set(table.columns))}"
        )
    domains = {subject: domain for domain, subjects in source.domains for subject in subjects}
    origins, subjects, exclusions, records = Counter(), Counter(), Counter(), []
    for index, row in enumerate(table.to_dict("records")):
        subject = str(row[subject_key])
        origin = str(row.get("src", f"{source.repository}/{subject}"))
        origins[origin] += 1
        subjects[subject] += 1
        if subject not in domains:
            exclusions["unselected_subject"] += 1
            continue
        if any(re.search(pattern, origin) for pattern in source.exclude_origins):
            exclusions["excluded_origin"] += 1
            continue
        question = row["question"]
        choices = list(row[choices_key])
        if isinstance(row[answer_key], bool) or not isinstance(row[answer_key], (int, np.integer)):
            raise ValueError("Answer indices must be integers")
        answer = int(row[answer_key])
        if (
            not isinstance(question, str)
            or not question.strip()
            or not 2 <= len(choices) <= 26
            or any(not isinstance(choice, str) for choice in choices)
            or not 0 <= answer < len(choices)
        ):
            raise ValueError(
                f"Malformed question/options/answer in {source.repository} row {index}"
            )
        text, answer_start = render_question(question, choices, answer)
        spaced, _ = render_question(question, choices, answer, spaced=True)
        if max(len(text.encode()), len(spaced.encode())) > max_bytes:
            exclusions["max_bytes"] += 1
            continue
        normalized = _normalize_question(question)
        if not normalized:
            raise ValueError("Empty normalized question")
        content = hashlib.sha256(normalized.encode()).hexdigest()
        row_id = str(row.get("question_id", index))
        records.append(
            {
                "id": f"{source.repository}:{row_id}",
                "domain": domains[subject],
                "source": source.repository,
                "revision": source.revision,
                "origin": origin,
                "subject": subject,
                "question": question,
                "choices": choices,
                "answer_index": answer,
                "text": text,
                "answer_start": answer_start,
                "group_id": content,
                "language": "en",
            }
        )
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Study source has duplicate observation IDs")
    return records, {
        **metadata,
        "rows": len(table),
        "columns": list(table.columns),
        "origins": dict(sorted(origins.items())),
        "subjects": dict(sorted(subjects.items())),
        "excluded": dict(exclusions),
        "eligible_by_domain": dict(Counter(row["domain"] for row in records)),
        "eligible_origins": dict(sorted(Counter(row["origin"] for row in records).items())),
        "domain_subjects": {domain: list(subjects) for domain, subjects in source.domains},
        "exclude_origin_patterns": list(source.exclude_origins),
    }


def inspect_sources(config, output=None):
    if not config.study.sources:
        raise ValueError("Configure study.sources before inspecting study data")
    report = {
        "kind": "local-study-source-audit",
        "max_bytes": config.study.max_bytes,
        "sources": [
            _read_source(source, config.study.max_bytes)[1] for source in config.study.sources
        ],
    }
    if output is not None:
        write_json(output, report)
    return report


def load_study_split(config, name):
    """Validate a frozen cohort before any model-dependent selection or outcome."""
    directory = config.paths.data / "study"
    manifest = read_json(directory / "manifest.json")
    if (
        manifest.get("schema") != 1
        or manifest.get("seed") != config.analysis.seed
        or manifest.get("max_bytes") != config.study.max_bytes
        or manifest.get("grouping", {}).get("threshold") != config.study.near_duplicate_threshold
    ):
        raise ValueError("Study cohort definition differs from the configured study")
    if name not in ("discovery", "calibration", "evaluation", "format_pairs"):
        raise ValueError("Unknown study partition")
    expected_sources = {
        (source.repository, source.revision, source.filename) for source in config.study.sources
    }
    actual_sources = {
        (source["repository"], source["revision"], source["filename"])
        for source in manifest["source_audits"]
    }
    if expected_sources != actual_sources:
        raise ValueError("Study source identities differ from the frozen cohort")
    audits = {audit["repository"]: audit for audit in manifest["source_audits"]}
    for source in config.study.sources:
        audit = audits[source.repository]
        required_columns = (
            {"subject", "choices", "answer"}
            if source.format == "mmlu"
            else {"category", "options", "answer_index"}
        )
        if (
            audit.get("domain_subjects")
            != {domain: list(subjects) for domain, subjects in source.domains}
            or audit.get("exclude_origin_patterns") != list(source.exclude_origins)
            or not required_columns.issubset(audit.get("columns", []))
        ):
            raise ValueError("Study source selection differs from the frozen cohort")
    for filename, expected in manifest["files"].items():
        path = Path(filename)
        if not path.is_relative_to(directory) or not path.is_file() or _hash(path) != expected:
            raise ValueError("Study partition bytes changed after freezing")
    groups = []
    cells = {
        (source.repository, domain)
        for source in config.study.sources
        for domain, _ in source.domains
    }
    for split, count in (
        ("discovery", config.study.discovery_per_cell),
        ("calibration", config.study.calibration_per_cell),
        ("evaluation", config.study.evaluation_per_cell),
    ):
        rows = read_json(directory / f"{split}.json")
        counts = Counter((row["source"], row["domain"]) for row in rows)
        if set(counts) != cells or any(value != count for value in counts.values()):
            raise ValueError("Frozen study cells differ from the declared sample counts")
        groups.append({row["group_id"] for row in rows})
    if any(first & second for index, first in enumerate(groups) for second in groups[index + 1 :]):
        raise ValueError("Study partitions share content groups")
    return read_json(directory / f"{name}.json")


def group_near_duplicates(records, threshold):
    """Conservatively group normalized question matches and word-shingle overlap.

    Grouping ignores choices, so augmented distractors cannot split an otherwise
    identical question. Textual grouping cannot guarantee semantic independence.
    """
    parent = list(range(len(records)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    inverted, sets, exact = defaultdict(list), [], {}
    merges = 0
    for index, record in enumerate(records):
        normalized = _normalize_question(record["question"])
        words = normalized.split()
        shingles = {tuple(words[start : start + 5]) for start in range(max(0, len(words) - 4))}
        candidates = Counter(candidate for shingle in shingles for candidate in inverted[shingle])
        if normalized in exact:
            candidates[exact[normalized]] = len(shingles)
        for candidate, intersection in candidates.items():
            union = len(shingles) + len(sets[candidate]) - intersection
            if normalized == _normalize_question(records[candidate]["question"]) or (
                union > 0 and intersection / union >= threshold
            ):
                first, second = root(index), root(candidate)
                if first != second:
                    parent[max(first, second)] = min(first, second)
                    merges += 1
        exact.setdefault(normalized, index)
        sets.append(shingles)
        for shingle in shingles:
            inverted[shingle].append(index)
    groups = defaultdict(list)
    for index in range(len(records)):
        groups[root(index)].append(index)
    output = []
    for indices in groups.values():
        group_id = min(records[index]["group_id"] for index in indices)
        output.extend({**records[index], "group_id": group_id} for index in indices)
    return output, {"groups": len(groups), "merged_records": merges, "threshold": threshold}


def partition_records(records, counts, *, seed):
    """Fill declared source/domain cells while keeping shared content in one split."""
    grouped = defaultdict(list)
    cells = sorted({(row["source"], row["domain"]) for row in records})
    for row in records:
        grouped[row["group_id"]].append(row)
    random = np.random.default_rng(seed)
    groups = sorted(grouped)
    random.shuffle(groups)
    splits = {name: [] for name in counts}
    used = {name: Counter() for name in counts}
    for group in groups:
        by_cell = {}
        for row in sorted(grouped[group], key=lambda row: row["id"]):
            by_cell.setdefault((row["source"], row["domain"]), row)
        scores = {
            name: sum(max(0, count - used[name][cell]) / count for cell in by_cell)
            for name, count in counts.items()
        }
        best = max(scores.values())
        if best <= 0:
            continue
        ties = [name for name in counts if scores[name] == best]
        name = ties[int(random.integers(len(ties)))]
        for cell, row in by_cell.items():
            if used[name][cell] < counts[name]:
                splits[name].append(row)
                used[name][cell] += 1
    shortages = {
        f"{name}:{source}:{domain}": counts[name] - used[name][(source, domain)]
        for name in counts
        for source, domain in cells
        if used[name][(source, domain)] < counts[name]
    }
    if shortages:
        raise ValueError(f"Insufficient independent content groups for declared cells: {shortages}")
    return splits


def prepare_study(config):
    """Prepare study partitions through the provenance-checked experiment runner."""
    inventory, records = [], []
    for source in config.study.sources:
        rows, audit = _read_source(source, config.study.max_bytes)
        inventory.append(audit)
        records.extend(rows)
        missing = set(dict(source.domains)) - {row["domain"] for row in rows}
        if missing:
            raise ValueError(f"Study source has empty declared domain cells: {sorted(missing)}")
    if not records:
        raise ValueError("No configured eligible study observations")
    grouped, grouping = group_near_duplicates(records, config.study.near_duplicate_threshold)
    group_domains = defaultdict(set)
    for row in grouped:
        group_domains[row["group_id"]].add(row["domain"])
    ambiguous = {group for group, domains in group_domains.items() if len(domains) > 1}
    grouping["ambiguous_domain_groups_excluded"] = len(ambiguous)
    grouped = [row for row in grouped if row["group_id"] not in ambiguous]
    splits = partition_records(
        grouped,
        {
            "discovery": config.study.discovery_per_cell,
            "calibration": config.study.calibration_per_cell,
            "evaluation": config.study.evaluation_per_cell,
        },
        seed=config.analysis.seed,
    )
    directory = config.paths.data / "study"
    outputs = []
    for name, rows in splits.items():
        path = directory / f"{name}.json"
        write_json(path, rows)
        outputs.append(path)
    pairs = []
    for row in splits["evaluation"]:
        spaced, start = render_question(
            row["question"], row["choices"], row["answer_index"], spaced=True
        )
        pairs.extend(
            [
                {**row, "variant": "standard"},
                {
                    **row,
                    "id": row["id"] + ":spaced",
                    "variant": "spaced",
                    "text": spaced,
                    "answer_start": start,
                },
            ]
        )
    pairs_path = directory / "format_pairs.json"
    write_json(pairs_path, pairs)
    outputs.append(pairs_path)
    manifest = {
        "schema": 1,
        "seed": config.analysis.seed,
        "source_audits": inventory,
        "grouping": grouping,
        "max_bytes": config.study.max_bytes,
        "splits": {
            name: {
                "documents": len(rows),
                "groups": len({row["group_id"] for row in rows}),
                "cells": dict(Counter(f"{row['source']}:{row['domain']}" for row in rows)),
            }
            for name, rows in splits.items()
        },
        "files": {str(path): _hash(path) for path in outputs},
        "limits": "Question-text grouping is not a guarantee against semantic or training-data overlap",
    }
    path = directory / "manifest.json"
    write_json(path, manifest)
    return [*outputs, path]
