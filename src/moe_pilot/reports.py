"""Offline interactive research reports and a lightweight localhost preview."""

import csv
import hashlib
import json
import math
import shutil
from datetime import UTC, datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def read_table(path):
    with path.open(newline="", encoding="utf-8") as stream:
        result = list(csv.DictReader(stream))
    for row in result:
        for key, value in row.items():
            if value == "":
                row[key] = None
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            if not math.isfinite(number):
                raise ValueError(f"Nonfinite value in report table {path}")
            row[key] = number
    return result


def study_snapshot(study):
    study = Path(study)
    result = {
        "progress": [],
        "study": {},
        "policy": None,
        "refreshed_at": datetime.now(UTC).isoformat(),
    }
    phases = {
        "discovery",
        "calibration",
        "evaluation",
        "evaluation-interventions",
        "evaluation-paired-chat",
    }
    for path in sorted(study.glob("*/*/manifest.json"), key=lambda path: path.stat().st_mtime):
        if path.parent.name not in phases:
            continue
        record = json.loads(path.read_text())
        chunks = record.get("chunks", [])
        result["policy"] = record.get("identity", {}).get("resources") or result["policy"]
        result["progress"].append(
            {
                "model": path.parent.parent.name,
                "partition": path.parent.name,
                "status": "complete" if record.get("status") == "complete" else "saved partial",
                "documents": record.get("documents", 0),
                "completed": sum(item.get("documents", 0) for item in chunks),
                "seconds": sum(item.get("seconds", 0) for item in chunks),
                "peak_gib": max((item.get("peak_process_bytes", 0) for item in chunks), default=0)
                / 1024**3,
                "resources": record.get("identity", {}).get("resources"),
                "job_budget_gib": record.get("identity", {})
                .get("resources", {})
                .get("max_memory_gib"),
                "reserve_gib": record.get("identity", {})
                .get("resources", {})
                .get("reserve_memory_gib"),
            }
        )
    if (study / "selection.json").exists():
        result["selection"] = json.loads((study / "selection.json").read_text())
    analysis = study / "tables" / "analysis_record.json"
    if analysis.exists():
        record = json.loads(analysis.read_text())
        if "table_sha256" not in record:
            raise ValueError("Rebuild study analysis before publishing its tables")
        for name, expected in record["table_sha256"].items():
            path = (study / "tables" / name).resolve()
            if (
                path.parent != (study / "tables").resolve()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected
            ):
                raise ValueError("Study analysis table changed after validation")
        for name in record["table_sha256"]:
            path = study / "tables" / name
            result["study"][path.stem] = read_table(path)
    return result


def build_report(reconstruction, study, output):
    """Publish reviewed CSV values into a self-contained, no-network HTML UI."""
    reconstruction, study, output = map(
        lambda path: Path(path).resolve(), (reconstruction, study, output)
    )
    if (
        output == reconstruction
        or reconstruction.is_relative_to(output)
        or study.is_relative_to(output)
        or output.is_relative_to(reconstruction)
        or output.is_relative_to(study)
    ):
        raise ValueError("Report output must be separate from research inputs")
    evidence = json.loads((reconstruction / "evidence.json").read_text())
    for relative, expected in evidence["publication_files"].items():
        path = (reconstruction / relative).resolve()
        if (
            not path.is_relative_to(reconstruction)
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"Published research evidence changed: {relative}")
    tables = {
        "baseline": "baseline_by_domain.csv",
        "routing": "layer_domain_metrics.csv",
        "ablations": "ablations/ablation_summary.csv",
        "geometry": "activation_orthogonality.csv",
        "subtopics": "subtopics/layer_within_vs_between_jsd.csv",
    }
    payload = {
        name: read_table(reconstruction / "tables" / filename) for name, filename in tables.items()
    }
    payload["record"] = evidence["derived"]
    payload.update(study_snapshot(study))
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("tables", "figures"):
        shutil.copytree(
            reconstruction / directory,
            output / "data" / "reconstruction" / directory,
            dirs_exist_ok=True,
        )
    if (study / "tables").exists():
        shutil.copytree(study / "tables", output / "data" / "study", dirs_exist_ok=True)
    if (study / "figures").exists():
        shutil.copytree(study / "figures", output / "figures", dirs_exist_ok=True)
        manifest = json.loads((study / "figures" / "manifest.json").read_text())
        for item in manifest.values():
            for filename, expected in item["sources"].items():
                path = (study / "tables" / filename).resolve()
                if (
                    path.parent != (study / "tables").resolve()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != expected
                ):
                    raise ValueError("Study figure does not match its current source table")
            for filename, expected in item["files"].items():
                path = (study / "figures" / filename).resolve()
                if (
                    path.parent != (study / "figures").resolve()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != expected
                ):
                    raise ValueError("Exported study figure changed after validation")
        payload["figures"] = manifest
    for path in sorted(study.glob("*/*/manifest.json")):
        if path.parent.name in (
            "discovery",
            "calibration",
            "evaluation",
            "evaluation-interventions",
            "evaluation-paired-chat",
        ):
            destination = (
                output / "data" / "provenance" / path.parent.parent.name / path.parent.name
            )
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination / "manifest.json")
    for filename in ("plan.json", "format-plan.json", "selection.json"):
        if (study / filename).exists():
            shutil.copy2(study / filename, output / "data" / filename)
    for filename in ("README.md", "analysis.md"):
        if (study / filename).exists():
            shutil.copy2(study / filename, output / filename)
    shutil.copy2(
        reconstruction / "evidence.json", output / "data" / "reconstruction" / "evidence.json"
    )
    assets = Path(__file__).with_name("report_assets")
    for name in ("report.css", "report.js"):
        shutil.copy2(assets / name, output / name)
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = (assets / "index.html").read_text()
    (output / "index.html").write_text(
        template.replace("__RESEARCH_DATA__", encoded), encoding="utf-8"
    )
    return output / "index.html"


def serve_report(directory, port=8765, study=Path("runs/causal-study")):
    """Serve only the report directory on loopback; no research inference/imports."""
    from functools import partial

    directory = Path(directory).resolve()
    if not (directory / "index.html").is_file():
        raise ValueError("Build the report before starting its preview")

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/api/study":
                return super().do_GET()
            try:
                payload = json.dumps(study_snapshot(study), allow_nan=False).encode()
            except (OSError, ValueError):
                self.send_error(503, "Study tables are being updated or need validation")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(directory)))
    print(f"Research report: http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
