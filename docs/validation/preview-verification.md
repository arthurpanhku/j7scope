# Research-preview validation record

Date: 2026-10-28. Release: 0.2.0rc1.

- Direct local Python suite: 58 passing tests, including session event/recording
  isolation, auth/cookies/origins, queue rejection/recovery, generation/intervention
  serialization, malformed/oversized requests, streaming failure cleanup, and plugins.
- Trace gallery validation: passed.
- Pretrained model: 13 checks passed with SmolLM2-135M CPU float32 at the pinned
  revision. See [the machine-readable report](smollm2-135m-cpu.json). Includes actual
  authenticated JSON/SSE HTTP generation and recorded-gallery access, independent
  activation capture, and identity-patch invariance.
- Wheel and source metadata: `twine check` passed. A clean virtual environment with
  no torch installed started the CLI outside the repository, served bundled assets,
  recorded a generation, and replayed its trace index.
- React/TypeScript application: production build passed after dependency updates;
  npm audit reported zero vulnerabilities. The existing large Plotly bundle still
  produces a size warning.
- Browser: cookie login and authenticated live SSE updates were manually checked
  against a temporary local demo service; demo output remained labelled synthetic.
- Docker engine and CUDA were unavailable locally. A dedicated preview CI workflow
  builds/runs the demo container and repeats the model/package checks. GPU container
  performance and Qwen research-quality read-outs are not certified by these checks.

The local Dvalin 0.19.0 scan did **not** return a VERIFIED verdict. Its two high
findings identify `torch.nn.Module.eval()` as dynamic code execution; manual review
confirms these are PyTorch inference-mode methods, not calls to Python's `eval`.
Its wrapped test command also returned failure, while the complete direct test
command passed. The original evidence is preserved under `.dvalin/artifacts/` in
the developer checkout. No rule suppression or security-certification claim was
introduced. Static-analysis results are supplementary to the explicit checks above.

This is implementation validation for a research preview, not evidence of lens
convergence, semantic accuracy, or cross-language shared representations.
