# Changelog

## 0.2.0rc1 — 2026-10-28 · Research Preview

J7Scope is now a self-hosted LLM representation observatory and reproducible J-lens
research toolkit. Bilingual experiments are its first application.

- Installable `j7scope`, `j7scope-serve`, and `j7scope-validate` commands, wheel and
  source distributions with bundled viewer/vocabulary; optional HF dependencies.
- Local/remote deployment instructions, JSON configuration, Docker/Compose examples.
- Bearer authentication, ephemeral research sessions, browser cookie connection,
  per-session SSE/backlog/recording access, and restricted CORS.
- Bounded FIFO model admission shared by generation and intervention, request and
  connection limits, cooperative generation deadlines, generic client errors, and
  atomic trace/index publication. Failed generations are not complete artifacts.
- Installed backend plugins, HF model structure adapters, configurable fitting
  corpora/concept lexicons, trace provenance and preview flags.
- Pretrained SmolLM2-135M CPU checks, real-model HTTP generation/recording checks,
  independent activation capture and identity-patch validation; isolated wheel smoke.
- Tensor/tuple decoder-output compatibility in activation patching and chat-template
  compatibility with current Transformers.
- Compatible frontend dependency security updates.

Limits: text-only chat, no tool execution, single model/process, trusted lab scope.
The small-model validation verifies implementation paths; it is not a converged
lens, a semantic-quality evaluation, or evidence of bilingual representation sharing.
Qwen GPU performance and the GPU container profile need hardware validation.
Commercial text APIs without activations cannot supply J-space read-outs. The
preview is distributed through GitHub Releases; no PyPI publication is implied.
