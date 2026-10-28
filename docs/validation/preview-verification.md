# Research-preview validation record

Date: 2026-10-28. Release: 0.2.0rc2.

This release refreshes the README, installation examples, and distribution version.
Service and model behavior are unchanged from rc1. It remains a research preview.

- Clean release archive: 54 passing Python tests, including session isolation,
  authentication, cookies/origins, admission limits, failure cleanup, and plugins.
- Trace gallery validation: passed.
- Pretrained model: 13 checks passed with pinned SmolLM2-135M CPU float32. See
  [the fresh machine-readable report](smollm2-135m-cpu.json). Checks include independent
  activation capture, identity patching, authenticated JSON/SSE generation, and
  recorded-gallery access.
- Distribution metadata: wheel and source archive pass `twine check`. The installed
  wheel starts outside the repository without PyTorch, serves bundled assets, and
  records/replays a generation. Package and CLI versions both report `0.2.0rc2`.
- README relative links and the pinned HF-extra requirement syntax were checked.
  Published release files carry SHA-256 checksums. Direct GitHub installation is
  documented; no PyPI publication is implied.
- CI repeats Python tests, trace checks, real-model validation, package installation,
  demo-container startup/generation, and the existing web build/dependency audit.

CUDA and a local Docker daemon were unavailable. The demo container is checked in
GitHub CI; GPU performance, the GPU container profile, lens convergence, semantic
quality, and cross-language representation sharing are not established by these
checks. Text-only commercial APIs without activations cannot provide these read-outs.

The rc1 local Dvalin 0.19.0 run did not return VERIFIED: it flagged two PyTorch
`torch.nn.Module.eval()` calls as dynamic code execution, and its wrapped test
command failed while the direct suite passed. Those calls were manually reviewed
as inference-mode methods, not Python `eval`. This documentation/version release
does not claim a new Dvalin verdict or independent security certification. The
[rc1 validation record](https://github.com/arthurpanhku/j7scope/blob/v0.2.0rc1/docs/validation/preview-verification.md)
retains the original assessment.
