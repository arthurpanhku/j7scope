"""Build a Contrast Lab record for two existing Trace v1 directories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from j7scope.artifacts import write_json  # noqa: E402
from j7scope.contrast import build_contrast  # noqa: E402
from j7scope.data import load_parallel_pairs  # noqa: E402
from j7scope.rigor import build_lexicon  # noqa: E402
from j7scope.trace import read_trace  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_a")
    parser.add_argument("trace_b")
    parser.add_argument("--trace-root", type=Path, default=ROOT / "results" / "traces")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--mode",
        choices=("cross_language", "intervention", "correctness"),
        default="cross_language",
    )
    parser.add_argument("--position-map", type=Path)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    trace_a = read_trace(args.trace_root / args.trace_a)
    trace_b = read_trace(args.trace_root / args.trace_b)
    position_map = None
    if args.position_map:
        raw = json.loads(args.position_map.read_text(encoding="utf-8"))
        position_map = raw.get("position_map", raw) if isinstance(raw, dict) else raw

    pairs = load_parallel_pairs(args.data_dir) if args.data_dir.exists() else None
    lexicon = build_lexicon(pairs=pairs)
    contrast = build_contrast(
        trace_a,
        trace_b,
        lexicon,
        position_map=position_map,
        mode=args.mode,
        n_permutations=args.permutations,
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    align = {
        "parallel_group": (
            trace_a["manifest"].get("parallel_group")
            or trace_b["manifest"].get("parallel_group")
            or f"{args.trace_a}--{args.trace_b}"
        ),
        "members": {"a": args.trace_a, "b": args.trace_b},
        "position_map": contrast["position_map"],
        "pair_rigor": contrast["pair_rigor"],
        "contrast": contrast,
    }
    for trace_id in (args.trace_a, args.trace_b):
        write_json(args.trace_root / trace_id / "align.json", align)
    print(json.dumps({
        "trace_a": args.trace_a,
        "trace_b": args.trace_b,
        "observed": contrast["calibration"]["observed"],
        "sharedness": contrast["calibration"]["sharedness"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
