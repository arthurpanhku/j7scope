# j7scope-serve · Research Preview

The installable service for J7Scope's self-hosted LLM representation observatory.
It hosts a supported model, computes J-lens read-outs, and serves session-scoped
SSE, recorded traces, and a bundled viewer. Local and cloud-hosted weights use the
same backend; a plain remote chat API is insufficient for internal activations.

See the [installation and remote deployment manual](../../docs/research-preview.md),
[extension contracts](../../docs/extending.md), and [validation report](../../docs/validation/smollm2-135m-cpu.json).

```bash
# From a checkout:
pip install -e '.[hf]'
j7scope --config examples/smollm-cpu.json
# Or no PyTorch required for mock:
pip install -e .
j7scope --backend mock --record ./traces
```

Source-only stdlib mock remains available from this directory as
`python -m j7scope_serve --backend mock` (without recording).

| Endpoint | Access / purpose |
|---|---|
| `GET /health` | Public liveness only |
| `GET /ready` | Authenticated model readiness and capabilities |
| `GET /v1/models` | Authenticated loaded model |
| `POST /v1/chat/completions` | Authenticated text-chat subset, streaming or JSON |
| `GET /jspace/stream` | Authenticated session-specific SSE and backlog |
| `POST /jspace/interventions` | Authenticated serialized activation patch |
| `POST /jspace/sessions` | Operator creates a separate, expiring session credential |
| `GET /jspace/session` | Inspect own session identity |
| `POST /auth/login`, `/auth/logout` | Bearer-to-cookie browser connection / logout |
| `GET /traces/*` | Only the authenticated session's recordings |
| `GET /`, `/live.html`, `/connect.html`, etc. | Public static viewer assets |

Loopback without a key retains a single local workspace. Non-loopback binding
requires `J7SCOPE_API_KEY` with at least 32 ASCII characters. Remote researchers
share a session key between their text client and browser connection; the operator
key belongs to a separate operator workspace. Use SSH forwarding or HTTPS with
`--secure-cookie`, and one process per model/data directory.

The server serializes all model tasks, limits connections, viewers, body size, and
generation length, and returns 503 with Retry-After when its queue fills. Generation
timeouts are cooperative between forward passes. Model loading happens before the
CLI accepts requests. Failed/partial requests are not published as complete traces.
Recorded directories and gallery indexes are atomically published. Read-out method
and hardware provenance are retained; all new traces are marked `preview:true`.

The `mock` backend is synthetic DEMO data. The `hf` backend has real pretrained
SmolLM2 CPU pipeline validation. GPU/model-specific semantic quality requires further
validation. `zh` and `en` columns are script buckets, not independent proof of shared
cross-language representations. Tool calling and structured outputs are unsupported.
