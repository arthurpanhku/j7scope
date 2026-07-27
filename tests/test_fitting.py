from types import SimpleNamespace

import pytest
import torch
from torch import nn

import j7scope.fitting as fitting
from j7scope.fitting import (
    _Capture,
    _decoder_layers,
    JLens,
    exact_jacobian_for_prompt,
    jacobian_error,
    paper_jacobian_for_prompt,
)


class _Batch(dict):
    def to(self, device):
        return _Batch({key: value.to(device) for key, value in self.items()})


class _Tokenizer:
    def __call__(self, prompt, return_tensors):
        assert return_tensors == "pt"
        token_ids = [int(token) for token in prompt.split()]
        return _Batch(input_ids=torch.tensor([token_ids], dtype=torch.long))

    def decode(self, token_ids):
        return str(token_ids[0])


class _LinearDecoderLayer(nn.Module):
    def __init__(self, matrix, *, returns_tuple=False):
        super().__init__()
        self.register_buffer("matrix", torch.tensor(matrix, dtype=torch.float32))
        self.returns_tuple = returns_tuple

    def forward(self, hidden_states, *, gain):
        output = gain * (hidden_states @ self.matrix.T)
        return (output,) if self.returns_tuple else output


class _Core(nn.Module):
    def __init__(self, matrices):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, len(matrices[0]))
        self.layers = nn.ModuleList(
            [
                _LinearDecoderLayer(matrix, returns_tuple=(index == 1))
                for index, matrix in enumerate(matrices)
            ]
        )
        self.norm = nn.Identity()


class _TinyCausalLM(nn.Module):
    def __init__(self, matrices):
        super().__init__()
        width = len(matrices[0])
        self.config = SimpleNamespace(hidden_size=width)
        self.model = _Core(matrices)
        self.lm_head = nn.Linear(width, 8, bias=False)
        self.gain = 1.0
        self.last_batch_size = None
        self.eval().requires_grad_(False)

    @property
    def device(self):
        return self.model.embed_tokens.weight.device

    @property
    def dtype(self):
        return self.model.embed_tokens.weight.dtype

    def forward(self, input_ids, *, use_cache):
        assert use_cache is False
        self.last_batch_size = input_ids.shape[0]
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            output = layer(hidden, gain=self.gain)
            hidden = output[0] if isinstance(output, tuple) else output
        return SimpleNamespace(logits=self.lm_head(hidden))


@pytest.fixture
def linear_lens():
    matrices = [
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, -1.0]],
        [[1.0, 2.0, 0.0], [0.0, -1.0, 1.0], [0.5, 0.0, 1.0]],
        [[2.0, 0.0, 1.0], [-1.0, 1.0, 0.0], [0.0, 0.5, 1.0]],
    ]
    model = _TinyCausalLM(matrices)
    lens = JLens(model, _Tokenizer(), layer=0)
    expected = torch.tensor(matrices[2]) @ torch.tensor(matrices[1])
    return lens, expected


def test_exact_jacobian_matches_analytic_tail(linear_lens):
    lens, expected = linear_lens

    actual = exact_jacobian_for_prompt(
        lens.model, lens.tokenizer, "1 2 3", layer=0, position=-1
    )

    torch.testing.assert_close(actual, expected)
    assert jacobian_error(actual, expected) == {
        "relative_frobenius": 0.0,
        "cosine_similarity": pytest.approx(1.0),
        "max_absolute": 0.0,
    }


def test_exact_estimator_sets_mean_jacobian(linear_lens):
    lens, expected = linear_lens

    actual = lens.estimate_jacobian_exact(
        ["1 2", "3 4 5"], show_progress=False
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(lens.J, expected)


def test_stochastic_estimator_converges_to_exact(linear_lens):
    lens, expected = linear_lens

    estimate = lens.estimate_jacobian(
        ["1 2 3"], n_probes=5000, seed=7, show_progress=False
    )
    diagnostics = jacobian_error(estimate, expected)

    assert diagnostics["relative_frobenius"] < 0.06
    assert diagnostics["cosine_similarity"] > 0.998


def test_exact_jacobian_matches_full_graph_vjp_on_tiny_qwen():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        use_cache=False,
    )
    model = Qwen2ForCausalLM(config).eval().requires_grad_(False)
    tokenizer = _Tokenizer()
    prompt = "1 2 3 4"

    exact = exact_jacobian_for_prompt(
        model, tokenizer, prompt, layer=0, position=-1, chunk_size=1
    )

    source_capture, target_capture = _Capture(), _Capture()
    layers = _decoder_layers(model)
    handles = [
        layers[0].register_forward_hook(source_capture),
        layers[-1].register_forward_hook(target_capture),
    ]
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.enable_grad():
            model(**inputs, use_cache=False)
            rows = []
            for output_index in range(config.hidden_size):
                (gradient,) = torch.autograd.grad(
                    target_capture.value[0, -1, output_index].float(),
                    source_capture.value,
                    retain_graph=output_index < config.hidden_size - 1,
                )
                rows.append(gradient[0, -1, :])
        full_graph_vjp = torch.stack(rows).float().cpu()
    finally:
        for handle in handles:
            handle.remove()

    torch.testing.assert_close(exact, full_graph_vjp)


def test_paper_estimator_matches_scalar_vjps_on_tiny_qwen():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        use_cache=False,
    )
    model = Qwen2ForCausalLM(config).eval().requires_grad_(False)
    tokenizer = _Tokenizer()
    prompt = "1 2 3 4"
    valid_positions = torch.tensor([0, 1, 2])

    paper = paper_jacobian_for_prompt(
        model, tokenizer, prompt, layer=0, skip_first=0, dim_batch=3
    )

    source_capture, target_capture = _Capture(), _Capture()
    layers = _decoder_layers(model)
    handles = [
        layers[0].register_forward_hook(source_capture),
        layers[-1].register_forward_hook(target_capture),
    ]
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.enable_grad():
            model(**inputs, use_cache=False)
            rows = []
            for output_index in range(config.hidden_size):
                objective = target_capture.value[
                    0, valid_positions, output_index
                ].sum()
                (gradient,) = torch.autograd.grad(
                    objective,
                    source_capture.value,
                    retain_graph=output_index < config.hidden_size - 1,
                )
                rows.append(gradient[0, valid_positions, :].mean(dim=0))
        scalar_vjps = torch.stack(rows).float().cpu()
    finally:
        for handle in handles:
            handle.remove()

    torch.testing.assert_close(paper, scalar_vjps)


def test_paper_estimator_replicates_prompt_batch(linear_lens):
    lens, expected = linear_lens

    actual = paper_jacobian_for_prompt(
        lens.model,
        lens.tokenizer,
        "1 2 3",
        layer=0,
        skip_first=0,
        dim_batch=2,
    )

    assert lens.model.last_batch_size == 2
    torch.testing.assert_close(actual, expected)


def test_paper_estimator_resumes_atomic_checkpoint(
    linear_lens, monkeypatch, tmp_path
):
    lens, expected = linear_lens
    checkpoint = tmp_path / "paper-fit.checkpoint.pt"
    real_estimator = fitting.paper_jacobian_for_prompt
    calls = []

    def fail_second_prompt(*args, **kwargs):
        calls.append(args[2])
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        return real_estimator(*args, **kwargs)

    monkeypatch.setattr(
        fitting, "paper_jacobian_for_prompt", fail_second_prompt
    )
    prompts = ["1 2 3", "3 4 5"]
    with pytest.raises(RuntimeError, match="simulated interruption"):
        lens.estimate_jacobian_paper(
            prompts,
            skip_first=0,
            dim_batch=2,
            show_progress=False,
            checkpoint_path=checkpoint,
        )

    partial = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert partial["completed"] is False
    assert partial["next_index"] == 1
    assert partial["n_used"] == 1

    actual = lens.estimate_jacobian_paper(
        prompts,
        skip_first=0,
        dim_batch=2,
        show_progress=False,
        checkpoint_path=checkpoint,
    )
    completed = torch.load(checkpoint, map_location="cpu", weights_only=True)

    assert calls == ["1 2 3", "3 4 5", "3 4 5"]
    assert completed["completed"] is True
    assert completed["next_index"] == 2
    assert completed["n_used"] == 2
    torch.testing.assert_close(actual, expected)


def test_paper_checkpoint_rejects_different_corpus(linear_lens, tmp_path):
    lens, _ = linear_lens
    checkpoint = tmp_path / "paper-fit.checkpoint.pt"
    lens.estimate_jacobian_paper(
        ["1 2 3"],
        skip_first=0,
        dim_batch=2,
        show_progress=False,
        checkpoint_path=checkpoint,
    )

    with pytest.raises(ValueError, match="corpus_sha1"):
        lens.estimate_jacobian_paper(
            ["1 2 4"],
            skip_first=0,
            dim_batch=2,
            show_progress=False,
            checkpoint_path=checkpoint,
        )


def test_exact_jacobian_rejects_invalid_position_and_chunk(linear_lens):
    lens, _ = linear_lens

    with pytest.raises(IndexError, match="out of range"):
        exact_jacobian_for_prompt(
            lens.model, lens.tokenizer, "1 2", layer=0, position=2
        )
    with pytest.raises(ValueError, match="chunk_size"):
        exact_jacobian_for_prompt(
            lens.model, lens.tokenizer, "1 2", layer=0, chunk_size=0
        )


def test_paper_jacobian_rejects_too_short_prompt(linear_lens):
    lens, _ = linear_lens

    with pytest.raises(ValueError, match="prompt too short"):
        paper_jacobian_for_prompt(
            lens.model, lens.tokenizer, "1 2", layer=0, skip_first=1
        )


@pytest.mark.parametrize(
    "method",
    [
        "estimate_jacobian",
        "estimate_jacobian_exact",
        "estimate_jacobian_paper",
    ],
)
def test_estimators_reject_empty_prompts(linear_lens, method):
    lens, _ = linear_lens

    with pytest.raises(ValueError, match="at least one"):
        getattr(lens, method)([], show_progress=False)
