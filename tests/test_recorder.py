import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SERVE_APP = ROOT / "apps" / "serve"
if str(SERVE_APP) not in sys.path:
    sys.path.insert(0, str(SERVE_APP))

from j7scope_serve.backends import HFBackend, MockBackend  # noqa: E402
from j7scope_serve.protocol import bucket_readout, script_of  # noqa: E402
from j7scope_serve.recorder import build_lexicon, record_trace  # noqa: E402


def test_recorded_trace_includes_jacobian_estimator_provenance(tmp_path):
    backend = MockBackend()
    buffered = []
    for seq, step in enumerate(backend.generate([])):
        buffered.append(
            {
                "seq": seq,
                "ts_rel": seq / 10,
                "token": step.token,
                "token_script": script_of(step.token),
                "readout": bucket_readout(step.topk, per_lang=8),
            }
        )
        if len(buffered) == 3:
            break

    trace_dir = record_trace(
        tmp_path,
        backend=backend,
        prompt="test",
        buffered=buffered,
        lexicon=build_lexicon(ROOT, backend),
        trace_id="provenance-test",
    )
    manifest = json.loads(
        (trace_dir / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["jacobian"] == {
        "corpus_id": "mock-synthetic",
        "estimator": "synthetic",
        "n_probes": None,
        "n_prompts": 0,
        "position": None,
        "seed": None,
        "sha1": None,
    }


def test_hf_backend_auto_dtype_handles_t4_and_cpu():
    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_bf16_supported():
            return False

    class _Torch:
        cuda = _Cuda()
        bfloat16 = "bf16"
        float16 = "fp16"
        float32 = "fp32"

    backend = HFBackend(device="cuda", dtype="auto")
    assert backend._resolve_dtype(_Torch) == "fp16"

    backend.device = "cpu"
    assert backend._resolve_dtype(_Torch) == "fp32"


def test_hf_backend_loads_precomputed_paper_jacobian(tmp_path):
    path = tmp_path / "jacobian.pt"
    expected = torch.eye(3)
    torch.save(expected, path)
    backend = HFBackend(jacobian_path=str(path))
    metadata = {
        "configuration": {
            "model": backend.model_name,
            "layer": backend.layer,
        },
        "provenance": {
            "model_revision_resolved": "main",
            "corpus_sha1": "corpus-hash",
            "prompts_used": 1000,
            "estimator": "paper_replicated_batch_vjp",
        },
        "result": {"tensor_sha1": backend._tensor_sha1(expected)},
    }
    path.with_suffix(".json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    actual = backend._load_precomputed_jacobian()

    assert backend.jacobian_estimator == "paper_replicated_batch_vjp"
    assert backend.jacobian_corpus_id == "corpus-hash"
    assert backend.jacobian_n_prompts == 1000
    torch.testing.assert_close(actual, expected)


def test_hf_backend_rejects_precomputed_jacobian_without_metadata(tmp_path):
    path = tmp_path / "jacobian.pt"
    torch.save(torch.eye(3), path)
    backend = HFBackend(jacobian_path=str(path))

    with pytest.raises(FileNotFoundError, match="metadata"):
        backend._load_precomputed_jacobian()
