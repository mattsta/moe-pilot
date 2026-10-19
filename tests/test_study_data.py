import hashlib
from collections import Counter
from dataclasses import replace

import pandas as pd
import pytest

from moe_pilot.artifacts import read_json, write_json
from moe_pilot.config import Study, StudySource, load_experiment
from moe_pilot.experiment import ExperimentRunner
from moe_pilot.study_data import (
    group_near_duplicates,
    inspect_sources,
    partition_records,
    prepare_study,
    source_record_path,
)


def source_fixture(tmp_path, *, name="first", source_format="mmlu", origin="original"):
    path = tmp_path / name / "test.parquet"
    path.parent.mkdir(parents=True)
    rows = []
    for domain in ("computing", "mathematics"):
        for index in range(9):
            row = {
                "question": f"What is the meaning of unique{index} within {domain}?",
                "src": origin,
            }
            if source_format == "mmlu":
                row.update(subject=domain, choices=["first", "second"], answer=1)
            else:
                row.update(
                    category=domain,
                    options=["first", "second"],
                    answer_index=1,
                    question_id=f"{domain}-{index}",
                )
            rows.append(row)
    pd.DataFrame(rows).to_parquet(path)
    source = StudySource(
        f"example/{name}",
        "a" * 40,
        source_format,
        "test.parquet",
        path,
        {"computing": ("computing",), "mathematics": ("mathematics",)},
    )
    write_json(
        source_record_path(source),
        {
            "repository": source.repository,
            "revision": source.revision,
            "filename": source.filename,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    )
    return source


def test_source_balanced_grouped_splits_preserve_pairs_and_provenance(experiment, tmp_path):
    sources = (
        source_fixture(tmp_path),
        source_fixture(tmp_path, name="second", source_format="mmlu-pro"),
    )
    config = replace(experiment, study=Study(2, 2, 2, sources=sources))
    ExperimentRunner(config).run("prepare-study")
    directory = config.paths.data / "study"
    splits = {
        name: read_json(directory / f"{name}.json")
        for name in ("discovery", "calibration", "evaluation")
    }
    groups = []
    for rows in splits.values():
        assert set(Counter((row["source"], row["domain"]) for row in rows).values()) == {2}
        assert len(rows) == 8
        groups.append({row["group_id"] for row in rows})
        assert all(row["text"][row["answer_start"] :] == " B" for row in rows)
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    pairs = read_json(directory / "format_pairs.json")
    assert len(pairs) == 16
    for original, spaced in zip(pairs[::2], pairs[1::2], strict=True):
        assert original["group_id"] == spaced["group_id"]
        assert original["question"] == spaced["question"]
        assert original["choices"] == spaced["choices"]
        assert "\n\n" in spaced["text"]
    manifest = read_json(directory / "manifest.json")
    assert manifest["grouping"]["merged_records"] > 0
    before = (directory / "manifest.json").stat().st_mtime_ns
    ExperimentRunner(config).run("prepare-study")
    assert (directory / "manifest.json").stat().st_mtime_ns == before


def test_source_changes_and_revision_mismatches_are_rejected(experiment, tmp_path):
    source = source_fixture(tmp_path)
    config = replace(experiment, study=Study(1, 1, 1, sources=(source,)))
    record = read_json(source_record_path(source))
    write_json(source_record_path(source), {**record, "revision": "b" * 40})
    with pytest.raises(ValueError, match="identity changed"):
        inspect_sources(config)
    write_json(source_record_path(source), record)
    source.path.write_bytes(source.path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="identity changed"):
        prepare_study(config)


def test_origin_exclusion_is_explicit_in_source_audit(experiment, tmp_path):
    source = replace(
        source_fixture(tmp_path, origin="ori_mmlu-computing"), exclude_origins=("mmlu",)
    )
    config = replace(experiment, study=Study(sources=(source,)))
    audit = inspect_sources(config)["sources"][0]
    assert audit["excluded"]["excluded_origin"] == audit["rows"]
    assert not audit["eligible_by_domain"]
    with pytest.raises(ValueError, match="empty declared"):
        prepare_study(config)


def test_grouping_uses_question_identity_instead_of_augmented_choices():
    records = [
        {"question": "A ring contains how many identity elements?", "group_id": "first"},
        {"question": "A RING contains how many identity elements?", "group_id": "second"},
        {"question": "Calculate the orbital period of this planet.", "group_id": "third"},
    ]
    grouped, _ = group_near_duplicates(records, 0.8)
    assert grouped[0]["group_id"] == grouped[1]["group_id"]
    assert grouped[2]["group_id"] != grouped[0]["group_id"]


def test_declared_cell_sizes_do_not_silently_shrink():
    records = [{"source": "one", "domain": "a", "group_id": "same", "id": "1"}]
    with pytest.raises(ValueError, match="Insufficient independent"):
        partition_records(records, {"discovery": 2, "evaluation": 2}, seed=17)


def test_study_config_is_path_independent_and_deeply_immutable(config_file, tmp_path):
    config_file.write_text(
        config_file.read_text()
        + """
[study]
discovery_per_cell = 2
[[study.sources]]
repository = "example/data"
revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
format = "mmlu"
filename = "test.parquet"
path = "../raw/test.parquet"
[study.sources.domains]
computing = ["computer_science"]
"""
    )
    config = load_experiment(config_file)
    assert config.study.sources[0].path == tmp_path / "raw/test.parquet"
    assert config.study.sources[0].domains == (("computing", ("computer_science",)),)
    assert config.record()["study"]["sources"][0]["path"] == str(tmp_path / "raw/test.parquet")
