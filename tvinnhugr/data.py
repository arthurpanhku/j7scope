"""Loading the parallel zh/en probe corpus (see data/ and data/concept_taxonomy.md)."""

import json
from pathlib import Path


def load_prompts(path):
    """Load one probe file (data/probe_prompts_{zh,en}.jsonl) as a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_parallel_pairs(data_dir):
    """Return {id: {"zh": record, "en": record}}, enforcing strict pairing.

    Raises if any prompt id appears in only one language — the corpus is only
    meaningful as syntax-aligned pairs.
    """
    data_dir = Path(data_dir)
    zh = {r["id"]: r for r in load_prompts(data_dir / "probe_prompts_zh.jsonl")}
    en = {r["id"]: r for r in load_prompts(data_dir / "probe_prompts_en.jsonl")}
    unpaired = set(zh) ^ set(en)
    if unpaired:
        raise ValueError(f"unpaired prompt ids: {sorted(unpaired)}")
    return {i: {"zh": zh[i], "en": en[i]} for i in zh}
