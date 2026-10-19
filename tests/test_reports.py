import hashlib
import json
import re

import pytest

from moe_pilot.artifacts import write_json
from moe_pilot.reports import build_report, study_snapshot


@pytest.fixture
def publication(tmp_path):
    root = tmp_path / "reconstruction"
    files = {
        "baseline_by_domain.csv": "domain,documents,valid_targets,mean_loss_nats_per_token\nscience,2,6,1.25\n",
        "layer_domain_metrics.csv": "domain,layer,jsd_vs_overall,jsd_null_mean\nscience,7,0.1,0.002\n",
        "ablations/ablation_summary.csv": "domain,scope,expert_count,control_family,paired_difference,ci95_low,ci95_high\nscience,layer-7,1,on-domain-control,0.1,0.01,0.2\n",
        "activation_orthogonality.csv": "layer,mean_abs_cos_centered\n7,0.02\n",
        "subtopics/layer_within_vs_between_jsd.csv": "domain,layer,within_domain_pair_jsd\nscience,7,0.03\n",
    }
    inventory = {}
    for name, value in files.items():
        path = root / "tables" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
        inventory[f"tables/{name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "figures").mkdir()
    write_json(
        root / "evidence.json",
        {
            "derived": {
                "documents": 2,
                "valid_targets": 6,
                "baseline_mean_loss": 1.25,
                "conditions": 3,
            },
            "publication_files": inventory,
        },
    )
    return root


def test_report_embeds_verified_tables_and_runs_without_external_scripts(publication, tmp_path):
    path = build_report(publication, tmp_path / "study", tmp_path / "report")
    text = path.read_text()
    match = re.search(
        r'<script id="research-data" type="application/json">(.*?)</script>', text, re.S
    )
    assert match
    data = json.loads(match.group(1))
    assert data["baseline"][0]["mean_loss_nats_per_token"] == 1.25
    assert data["record"]["documents"] == 2
    assert 'src="http' not in text
    assert (path.parent / "report.js").is_file()
    assert (path.parent / "data/reconstruction/tables/baseline_by_domain.csv").is_file()


def test_tampered_published_inputs_are_not_rendered(publication, tmp_path):
    path = publication / "tables/baseline_by_domain.csv"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="evidence changed"):
        build_report(publication, tmp_path / "study", tmp_path / "report")


def test_progress_excludes_admission_checks_and_preserves_actual_policy(tmp_path):
    record = {
        "documents": 180,
        "status": "running",
        "identity": {"resources": {"max_memory_gib": 5, "reserve_memory_gib": 16}},
        "chunks": [{"documents": 16, "seconds": 12, "peak_process_bytes": 1024**3}],
    }
    for phase in ("evaluation", "discovery-admission-v1"):
        write_json(tmp_path / "model" / phase / "manifest.json", record)
    snapshot = study_snapshot(tmp_path)
    assert len(snapshot["progress"]) == 1
    assert snapshot["progress"][0]["completed"] == 16
    assert snapshot["progress"][0]["status"] == "saved partial"
    assert snapshot["policy"]["max_memory_gib"] == 5


def test_unvalidated_or_tampered_study_tables_are_not_displayed(tmp_path):
    path = tmp_path / "tables" / "summary.csv"
    path.parent.mkdir()
    path.write_text("value\n1\n")
    assert study_snapshot(tmp_path)["study"] == {}
    write_json(
        path.parent / "analysis_record.json",
        {"table_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}},
    )
    assert study_snapshot(tmp_path)["study"]["summary"] == [{"value": 1}]
    path.write_text("value\n2\n")
    with pytest.raises(ValueError, match="changed after validation"):
        study_snapshot(tmp_path)


def test_report_output_cannot_overwrite_or_mix_with_research_inputs(publication, tmp_path):
    with pytest.raises(ValueError, match="separate"):
        build_report(publication, tmp_path / "study", publication / "report")


def test_report_rejects_observation_provenance_changed_after_analysis(tmp_path):
    path = tmp_path / "model" / "evaluation" / "manifest.json"
    write_json(path, {"status": "complete"})
    write_json(
        tmp_path / "tables" / "analysis_record.json",
        {
            "table_sha256": {},
            "observation_manifests": {
                str(path.relative_to(tmp_path)): hashlib.sha256(path.read_bytes()).hexdigest()
            },
        },
    )
    study_snapshot(tmp_path)
    write_json(path, {"status": "changed"})
    with pytest.raises(ValueError, match="observations changed"):
        study_snapshot(tmp_path)
