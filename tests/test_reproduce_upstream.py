import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "reproduce_upstream.py"


def test_upstream_reproduction_dry_run_needs_no_gpu_or_downloads():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    report = json.loads(completed.stdout)

    assert report["status"] == "dry_run"
    config = report["configuration"]
    assert config["model"] == "Qwen/Qwen3.5-4B"
    assert config["lens_repo"] == "neuronpedia/jacobian-lens"
    assert config["lens_revision_requested"] == "qwen-n1000"
    assert config["position"] == -2
    assert config["requires_cuda"] is True
    assert len(config["upstream_commit"]) == 40
