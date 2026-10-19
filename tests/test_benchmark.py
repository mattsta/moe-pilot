import json

import pytest
import torch

from moe_pilot.benchmark import benchmark_engine
from moe_pilot.cli import main


def test_default_benchmark_never_constructs_an_extra_cpu_reference(local_model, monkeypatch):
    from moe_pilot import benchmark
    from moe_pilot.cli import parser

    _, source = local_model
    constructed = []
    original = benchmark.Engine

    def track(*args, **kwargs):
        constructed.append(kwargs["device"])
        assert kwargs["device"] != "cpu", "No implicit CPU reference engine"
        return original(*args, **kwargs)

    monkeypatch.setattr(benchmark, "Engine", track)
    report = benchmark_engine(source.store.directory, max_tokens=4, repeats=1, threads=1)
    assert len(constructed) == 1
    assert len(report["results"]) == 1
    assert report["method"]["correctness_reference"]["requested"] == "auto"
    args = parser().parse_args(["benchmark", "--output", "unused.json"])
    assert args.devices == ["auto"]
    assert args.threads == 2
    assert args.workloads == ["baseline"]


def test_benchmark_cli_records_real_cpu_runs_without_changing_threads(tmp_path, capsys):
    output = tmp_path / "measurement.json"
    previous = torch.get_num_threads()
    assert (
        main(
            [
                "benchmark",
                "--profile",
                "tiny",
                "--devices",
                "cpu",
                "auto",
                "--documents",
                "1",
                "--max-tokens",
                "4",
                "--warmup",
                "1",
                "--repeats",
                "2",
                "--threads",
                "1",
                "--output",
                str(output),
                "--workloads",
                "baseline",
                "routing_geometry",
                "ablations",
            ]
        )
        == 0
    )
    assert torch.get_num_threads() == previous
    report = json.loads(output.read_text())
    assert report["kind"] == "synthetic-engine-benchmark"
    assert report["model"]["synthetic_profile"] == "tiny"
    assert report["environment"]["threads"] == 1
    assert "benchmark.py" in report["implementation_sha256"]
    assert "Measurement record:" in capsys.readouterr().out
    cpu_rows = [row for row in report["results"] if row["backend"] == "cpu"]
    assert len(cpu_rows) == 3
    for row in report["results"]:
        assert len(row["seconds"]) == 2
        assert 0 < row["min_seconds"] <= row["median_seconds"] <= row["max_seconds"]
        assert row["valid_targets"] == 3
        assert row["max_loss_error_vs_reference"] < 2e-4
        assert row["speedup_vs_cpu"] > 0
    assert all(row["speedup_vs_cpu"] == 1 for row in cpu_rows)


def test_local_model_benchmark_is_distinguished_from_synthetic_fixture(local_model):
    _, engine = local_model
    report = benchmark_engine(
        engine.store.directory,
        devices=("cpu",),
        document_counts=(1,),
        max_tokens=4,
        warmup=1,
        repeats=1,
        threads=1,
    )
    assert report["kind"] == "local-model-engine-benchmark"
    assert report["model"]["synthetic_profile"] is None
    assert report["model"]["local_fingerprint"] == engine.store.fingerprint()


def test_failed_benchmark_restores_threads_and_does_not_publish_partial_report(tmp_path):
    previous = torch.get_num_threads()
    output = tmp_path / "failed.json"
    with pytest.raises(FileNotFoundError):
        benchmark_engine(tmp_path / "absent", threads=1, output=output)
    assert torch.get_num_threads() == previous
    assert not output.exists()


@pytest.mark.parametrize("options", [{"repeats": 0}, {"warmup": 0}, {"document_counts": (0,)}])
def test_invalid_measurements_fail_before_creating_output(tmp_path, options):
    output = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        benchmark_engine(output=output, **options)
    assert not output.exists()
