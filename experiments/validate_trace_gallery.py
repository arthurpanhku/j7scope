"""Validate a Trace v1 gallery and frontend rigor-layer boundaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from j7scope.trace import validate_trace_gallery  # noqa: E402


FORBIDDEN_FRONTEND_FORMULAS = (
    "cross_lang_overlap -",
    "same_lang_baseline -",
    "(obs - null",
    "(obs-null",
)


def validate_frontend_boundary(repo_root: Path) -> list[str]:
    """Ensure executable frontend code displays, but does not derive, sharedness."""
    problems = []
    source_roots = (repo_root / "apps" / "site", repo_root / "apps" / "web" / "src")
    for source_root in source_roots:
        if not source_root.exists():
            continue
        for path in source_root.rglob("*"):
            if path.suffix not in {".js", ".ts", ".tsx"}:
                continue
            text = path.read_text(encoding="utf-8")
            for formula in FORBIDDEN_FRONTEND_FORMULAS:
                if formula in text:
                    problems.append(
                        f"{path.relative_to(repo_root)} recomputes rigor formula "
                        f"({formula!r}); frontend must display trace values only"
                    )
    return problems


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trace_root",
        nargs="?",
        type=Path,
        default=ROOT / "results" / "traces",
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)

    problems = validate_trace_gallery(args.trace_root)
    problems.extend(validate_frontend_boundary(args.repo_root))
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Validated trace gallery: {args.trace_root}")


if __name__ == "__main__":
    main()
