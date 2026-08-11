from copy import deepcopy

from j7scope.contrast import (
    PAIR_SHAREDNESS_DEFINITION,
    build_contrast,
    detect_answer_identity_confound,
)


def _token(seq, concept, surface):
    return {
        "seq": seq,
        "token": surface,
        "readout": {
            "zh": [{"token": f"中-{surface}", "rank": 0, "score": 4.0,
                    "concept": concept}],
            "en": [{"token": f"en-{surface}", "rank": 1, "score": 3.0,
                    "concept": concept}],
            "other": [],
        },
    }


def _trace(trace_id, language, concepts, **manifest_fields):
    manifest = {
        "trace_id": trace_id,
        "model": "test-model",
        "revision": "abc123",
        "layer": 12,
        "language": language,
        "jacobian": {"sha1": "lens-hash", "corpus_id": "corpus-hash",
                     "estimator": "test"},
        "capture": {"tool": "pytest", "created_at": "2026-01-01T00:00:00Z"},
        "is_demo": False,
    }
    manifest.update(manifest_fields)
    return {
        "manifest": manifest,
        "tokens": [_token(i, concept, f"t{i}") for i, concept in enumerate(concepts)],
        "metrics": {},
    }


def test_build_contrast_is_deterministic_and_bakes_research_controls():
    trace_a = _trace("a", "en", ["alpha", "beta", "gamma", "delta"])
    trace_b = _trace("b", "zh", ["alpha", "beta", "gamma", "delta"])

    first = build_contrast(
        trace_a, trace_b, {}, n_permutations=200, n_boot=200, seed=9
    )
    second = build_contrast(
        trace_a, trace_b, {}, n_permutations=200, n_boot=200, seed=9
    )

    assert first == second
    assert first["calibration"]["observed"] == 1.0
    assert first["calibration"]["null"]["n"] == 200
    assert first["calibration"]["sharedness"]["definition"] == PAIR_SHAREDNESS_DEFINITION
    assert first["concept_trajectories"][0]["rank_a"][0] in (1, None)
    assert first["provenance"]["trace_a"]["lens_sha1"] == "lens-hash"
    assert len(first["pair_rigor"]) == 4


def test_answer_identity_check_flags_and_clears_confound():
    base = _trace("a", "en", ["alpha"], correct=True, answer_identity="A")
    changed_answer = _trace("b", "en", ["alpha"], correct=False, answer_identity="B")
    same_answer = deepcopy(changed_answer)
    same_answer["manifest"]["answer_identity"] = "A"

    confounded = detect_answer_identity_confound(base["manifest"], changed_answer["manifest"])
    controlled = detect_answer_identity_confound(base["manifest"], same_answer["manifest"])

    assert confounded["status"] == "confounded"
    assert confounded["controlled"] is False
    assert controlled["status"] == "controlled"
    assert controlled["controlled"] is True


def test_answer_identity_check_never_guesses_from_free_text():
    result = detect_answer_identity_confound({"prompt": "correct"}, {"prompt": "wrong"})
    assert result["status"] == "not_assessable"
    assert result["controlled"] is None
