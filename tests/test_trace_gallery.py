import json
import subprocess
import sys
from pathlib import Path

from j7scope.trace import validate_trace_gallery


ROOT = Path(__file__).resolve().parents[1]


def test_demo_builder_produces_valid_gallery(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments" / "build_demo_trace.py"),
            "--out",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert validate_trace_gallery(tmp_path) == []
    align = json.loads(
        (tmp_path / "demo-parallel-en" / "align.json").read_text(encoding="utf-8")
    )
    assert align["contrast"]["calibration"]["null"]["n"] == 1000
    assert align["contrast"]["concept_trajectories"]
    assert align["contrast"]["answer_identity_check"]["status"] == "not_assessable"


def test_contrast_lab_page_keeps_research_statistics_precomputed():
    page = (ROOT / "apps" / "site" / "contrast.html").read_text(encoding="utf-8")

    assert "Contrast Lab" in page
    assert "C.calibration" in page
    assert "jspace/interventions" in page
    assert "j7scope.claim_card" in page
    assert "cross_lang_overlap -" not in page


def test_gallery_validator_detects_index_drift(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments" / "build_demo_trace.py"),
            "--out",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    index_path = tmp_path / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["traces"][0]["n_tokens"] += 1
    index_path.write_text(json.dumps(index), encoding="utf-8")

    problems = validate_trace_gallery(tmp_path)

    assert any("index.n_tokens does not match" in problem for problem in problems)


def test_capture_cli_dry_run_needs_no_gpu():
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments" / "capture_trace.py"),
            "--trace-id",
            "community-deception-en",
            "--language",
            "en",
            "--concept",
            "deception",
            "--prompt",
            "Explain deception briefly.",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    config = json.loads(completed.stdout)

    assert config["model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config["dtype"] == "auto"
    assert config["trace_id"] == "community-deception-en"
