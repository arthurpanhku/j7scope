"""Build a deterministic demo artifact run for the J-Space Explorer.

This script does not run a model. It creates small, clearly marked demo data
from the existing probe corpus so frontend development can proceed before GPU
jobs produce real J-lens and activation-patching results.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from j7scope.artifacts import RUN_FILES, write_json, write_run
from j7scope.data import load_parallel_pairs


CATEGORY_ORDER = [
    "deception",
    "manipulation",
    "concession",
    "eval_awareness",
    "emotion",
    "multihop",
    "entity",
]

CATEGORY_BASE = {
    "deception": (-3.4, 1.6),
    "manipulation": (-2.2, -0.8),
    "concession": (-0.8, 2.2),
    "eval_awareness": (0.6, 0.9),
    "emotion": (2.0, -0.2),
    "multihop": (3.2, 1.7),
    "entity": (3.6, -1.6),
}

FILLER_TOKENS = {
    "zh": ["概念", "意图", "判断", "行为", "结果"],
    "en": ["concept", "intent", "judgment", "behavior", "result"],
}


def _score(seed: int, lo: float, hi: float) -> float:
    wave = math.sin(seed * 12.9898) * 43758.5453
    frac = wave - math.floor(wave)
    return round(lo + frac * (hi - lo), 3)


def _readout(record: dict, lang: str, hit: bool = True) -> list[dict]:
    expected = list(record["expected"])
    tokens = expected if hit else FILLER_TOKENS[lang]
    fillers = [t for t in FILLER_TOKENS[lang] if t not in tokens]
    merged = (tokens + fillers)[:8]
    return [
        {
            "rank": i + 1,
            "token": token,
            "score": round(12.0 - i * 0.74 - (0.0 if hit else 2.9), 3),
            "is_expected": token in expected,
        }
        for i, token in enumerate(merged)
    ]


def _point(pair_id: str, record: dict, lang: str, index: int, condition: str) -> dict:
    bx, by = CATEGORY_BASE[record["category"]]
    lang_shift = -0.16 if lang == "zh" else 0.16
    condition_shift = 0.22 if condition == "patched" else 0.0
    jitter_x = _score(index + (0 if lang == "zh" else 21), -0.22, 0.22)
    jitter_y = _score(index + (7 if lang == "zh" else 31), -0.22, 0.22)
    return {
        "id": f"{pair_id}:{lang}:{condition}",
        "pair_id": pair_id,
        "language": lang,
        "category": record["category"],
        "concept": record["concept"],
        "abstractness": record["abstractness"],
        "condition": condition,
        "x": round(bx + lang_shift + condition_shift + jitter_x, 3),
        "y": round(by - lang_shift * 0.6 + condition_shift * 0.4 + jitter_y, 3),
    }


def _transport_success(record: dict, source_lang: str, target_lang: str, layer: int, index: int) -> bool:
    if source_lang == target_lang:
        return True
    if record["abstractness"] == "abstract":
        return layer >= 10 or index % 3 != 0
    return layer >= 14 and index % 4 == 0


def build_demo(data_dir: Path, run_id: str) -> tuple[dict, list[dict], list[dict], dict, dict, dict]:
    pairs = load_parallel_pairs(data_dir)
    ordered = sorted(pairs.items(), key=lambda item: (CATEGORY_ORDER.index(item[1]["en"]["category"]), item[0]))
    layers = [6, 8, 10, 12, 14, 16, 18]
    jlens_layer = 18
    patch_layers = [8, 10, 12, 14]

    readouts = []
    points = []
    links = []
    patches = []

    for index, (pair_id, pair) in enumerate(ordered):
        for lang in ("zh", "en"):
            record = pair[lang]
            readouts.append(
                {
                    "run_id": run_id,
                    "pair_id": pair_id,
                    "language": lang,
                    "category": record["category"],
                    "concept": record["concept"],
                    "abstractness": record["abstractness"],
                    "prompt": record["text"],
                    "expected": record["expected"],
                    "jlens_layer": jlens_layer,
                    "position": -1,
                    "topk": _readout(record, lang, hit=True),
                    "concept_hit": True,
                }
            )
            points.append(_point(pair_id, record, lang, index, "baseline"))
            if record["abstractness"] == "abstract":
                points.append(_point(pair_id, record, lang, index + 100, "patched"))

        links.append(
            {
                "source": f"{pair_id}:zh:baseline",
                "target": f"{pair_id}:en:baseline",
                "pair_id": pair_id,
                "kind": "translation_pair",
            }
        )

        for patch_layer in patch_layers:
            for source_lang, target_lang in (("zh", "en"), ("en", "zh"), ("zh", "zh"), ("en", "en")):
                source = pair[source_lang]
                target = pair[target_lang]
                success = _transport_success(source, source_lang, target_lang, patch_layer, index)
                cross = source_lang != target_lang
                leakage = _score(index + patch_layer + (11 if source_lang == "zh" else 19), 0.02, 0.16)
                if not cross:
                    leakage = _score(index + patch_layer, 0.0, 0.04)
                patches.append(
                    {
                        "run_id": run_id,
                        "patch_id": f"{pair_id}:{source_lang}->{target_lang}:L{patch_layer}:concept",
                        "pair_id": pair_id,
                        "source_language": source_lang,
                        "target_language": target_lang,
                        "source_concept": source["concept"],
                        "target_context_concept": target["concept"],
                        "category": source["category"],
                        "abstractness": source["abstractness"],
                        "patch_layer": patch_layer,
                        "jlens_layer": jlens_layer,
                        "control_type": "cross_language_concept" if cross else "same_language_concept",
                        "transport_success": success,
                        "language_preserved": leakage < 0.18,
                        "concept_score": round((0.78 if success else 0.34) + patch_layer * 0.01, 3),
                        "source_language_leakage": leakage,
                        "null_gap": round((0.42 if success else 0.08) + patch_layer * 0.006, 3),
                        "readout": _readout(target, target_lang, hit=success),
                        "next_token": _readout(target, target_lang, hit=success)[:5],
                    }
                )

            for control_type in ("random_same_norm", "unrelated_concept"):
                for source_lang, target_lang in (("zh", "en"), ("en", "zh")):
                    source = pair[source_lang]
                    target = pair[target_lang]
                    patches.append(
                        {
                            "run_id": run_id,
                            "patch_id": f"{pair_id}:{source_lang}->{target_lang}:L{patch_layer}:{control_type}",
                            "pair_id": pair_id,
                            "source_language": source_lang,
                            "target_language": target_lang,
                            "source_concept": source["concept"],
                            "target_context_concept": target["concept"],
                            "category": source["category"],
                            "abstractness": source["abstractness"],
                            "patch_layer": patch_layer,
                            "jlens_layer": jlens_layer,
                            "control_type": control_type,
                            "transport_success": False,
                            "language_preserved": True,
                            "concept_score": _score(index + patch_layer, 0.08, 0.28),
                            "source_language_leakage": _score(index + patch_layer, 0.0, 0.06),
                            "null_gap": _score(index + patch_layer, -0.03, 0.08),
                            "readout": _readout(target, target_lang, hit=False),
                            "next_token": _readout(target, target_lang, hit=False)[:5],
                        }
                    )

    layer_rows = []
    for layer in layers:
        abstract_rate = min(0.92, max(0.28, 0.18 + layer * 0.044))
        concrete_rate = min(0.56, max(0.08, -0.18 + layer * 0.038))
        layer_rows.append(
            {
                "layer": layer,
                "cross_language_success": round(abstract_rate, 3),
                "concrete_success": round(concrete_rate, 3),
                "same_language_success": round(min(0.96, abstract_rate + 0.08), 3),
                "null_success": round(max(0.03, abstract_rate - 0.46), 3),
                "source_language_leakage": round(max(0.03, 0.2 - layer * 0.007), 3),
            }
        )

    cross_patches = [p for p in patches if p["control_type"] == "cross_language_concept" and p["patch_layer"] == 12]
    null_patches = [p for p in patches if p["control_type"] != "cross_language_concept" and p["patch_layer"] == 12]
    abstract_patches = [p for p in cross_patches if p["abstractness"] == "abstract"]
    concrete_patches = [p for p in cross_patches if p["abstractness"] == "concrete"]

    def rate(rows: list[dict], key: str) -> float:
        return round(sum(1 for row in rows if row[key]) / max(1, len(rows)), 3)

    manifest = {
        "run_id": run_id,
        "label": "Demo Qwen2.5-7B layer 18",
        "is_demo": True,
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "status": "demo_artifact",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "Synthetic, deterministic data for frontend development. Replace with real activation patching artifacts before analysis.",
        "jlens_layer": jlens_layer,
        "patch_layers": patch_layers,
        "languages": ["zh", "en"],
        "categories": CATEGORY_ORDER,
        "artifact_version": 1,
        "files": RUN_FILES,
    }

    layer_scan = {
        "run_id": run_id,
        "model": manifest["model"],
        "metric": "transport_success_rate",
        "rows": layer_rows,
    }

    projections = {
        "run_id": run_id,
        "basis": "demo_deterministic_layout",
        "description": "Python-side projection placeholder for frontend development.",
        "points": points,
        "links": links,
    }

    metrics = {
        "run_id": run_id,
        "primary_patch_layer": 12,
        "summary": {
            "pairs": len(ordered),
            "readouts": len(readouts),
            "patches": len(patches),
            "cross_language_success": rate(cross_patches, "transport_success"),
            "abstract_cross_language_success": rate(abstract_patches, "transport_success"),
            "concrete_cross_language_success": rate(concrete_patches, "transport_success"),
            "null_success": rate(null_patches, "transport_success"),
            "language_preservation": rate(cross_patches, "language_preserved"),
            "mean_null_gap": round(sum(p["null_gap"] for p in cross_patches) / max(1, len(cross_patches)), 3),
        },
    }

    return manifest, readouts, patches, layer_scan, projections, metrics


def copy_run(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for filename in RUN_FILES.values():
        shutil.copy2(src / filename, dest / filename)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results/runs/demo-qwen25-7b-l18"))
    parser.add_argument("--public-out", type=Path, default=Path("apps/web/public/runs/demo-qwen25-7b-l18"))
    parser.add_argument("--run-id", default="demo-qwen25-7b-l18")
    args = parser.parse_args()

    manifest, readouts, patches, layer_scan, projections, metrics = build_demo(args.data_dir, args.run_id)
    write_run(
        args.out,
        manifest=manifest,
        readouts=readouts,
        patches=patches,
        layer_scan=layer_scan,
        projections=projections,
        metrics=metrics,
    )
    copy_run(args.out, args.public_out)
    write_json(args.public_out.parent / "index.json", [{"run_id": args.run_id, "label": manifest["label"], "path": args.run_id}])
    print(f"Wrote demo run to {args.out} and {args.public_out}")


if __name__ == "__main__":
    main()
