import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fit_paper_jacobian import (
    DEFAULT_CORPUS,
    _configuration,
    _load_local_prompts,
    parse_args,
)


def test_default_smoke_corpus_is_valid():
    prompts = _load_local_prompts(
        DEFAULT_CORPUS, text_key="text", max_prompts=1000, min_chars=0
    )

    assert len(prompts) == 12
    assert all(len(prompt) > 40 for prompt in prompts)


def test_local_corpus_reports_invalid_jsonl(tmp_path):
    corpus = tmp_path / "broken.jsonl"
    corpus.write_text('{"text": "ok"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        _load_local_prompts(
            corpus, text_key="text", max_prompts=10, min_chars=0
        )


def test_dry_run_configuration_marks_default_as_smoke():
    args = parse_args(["--dry-run"])
    configuration = _configuration(args)

    assert configuration["corpus"]["smoke_only"] is True
    assert configuration["resume"] is True
    json.dumps(configuration)
