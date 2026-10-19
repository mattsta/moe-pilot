"""The checked-in reader report must work from a fresh checkout, offline."""

import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


def test_checked_in_report_includes_local_links_tables_and_provenance():
    root = Path(__file__).resolve().parents[1] / "research" / "report"

    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key not in ("href", "src") or not value:
                    continue
                target = urlsplit(value)
                if target.scheme or target.netloc or not target.path:
                    continue
                assert (root / unquote(target.path)).is_file(), value

    html = (root / "index.html").read_text(encoding="utf-8")
    Links().feed(html)
    start = html.index('type="application/json">') + len('type="application/json">')
    payload = json.loads(html[start : html.index("</script>", start)])
    for name in payload["study"]:
        assert (root / "data" / "study" / f"{name}.csv").is_file()
    record = json.loads((root / "data" / "study" / "analysis_record.json").read_text())
    for relative, expected in record["observation_manifests"].items():
        path = root / "data" / "provenance" / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    for item in payload["figures"].values():
        for filename, expected in item["files"].items():
            assert (
                hashlib.sha256((root / "figures" / filename).read_bytes()).hexdigest() == expected
            )
