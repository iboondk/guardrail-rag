"""End-to-end check of the eval runner: exit code, report files and leak count."""

from __future__ import annotations

import json
from pathlib import Path

from guardrail_rag.eval.run import main


def test_eval_run_writes_reports_and_has_no_leaks(tmp_path: Path) -> None:
    md_path = tmp_path / "r.md"
    json_path = tmp_path / "r.json"

    exit_code = main(["--out", str(md_path), "--json", str(json_path)])

    assert exit_code == 0
    assert md_path.is_file()
    assert json_path.is_file()

    md = md_path.read_text(encoding="utf-8")
    assert "| Metric |" in md
    assert "Cross-tenant leaks" in md

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["leaks"]["total_findings"] == 0
    assert report["injection"]["pass_rate"] == 1.0
    assert report["pii"]["answers_with_pii"] == 0
    assert len(report["retrieval"]["ablation"]) == 4
