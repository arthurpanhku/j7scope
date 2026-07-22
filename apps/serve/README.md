# j7scope-serve · Live J-Space sidecar

A side-channel that shows the **J-space read-out of an open-weights model in
real time**, while an agent harness (opencode, codex, …) is driving that model.

```
┌──────────────┐   OpenAI /v1/chat/completions    ┌────────────────────────┐
│  opencode /  │ ───────────────────────────────▶ │  j7scope-serve        │
│  codex /     │ ◀──────── token stream ───────── │  (proxy + J-lens hook)  │
│  any OpenAI  │                                   │                         │
│  client      │                                   │  per token: read out    │
└──────────────┘                                   │  h_l ─J_l─▶ concepts    │
       ▲                                            └───────────┬────────────┘
       │ coding as usual                                        │ SSE side-channel
┌──────┴───────┐        GET /jspace/stream                      │ {token, zh[], en[]}
│ J-Space view │ ◀──────────────────────────────────────────────┘
│  (browser)   │
└──────────────┘
```

**Why only open models.** The read-out is computed from the model's residual
stream (`h_l`), so the sidecar must host the weights itself. Closed API models
(Anthropic, OpenAI, …) never expose activations — for those, the harness works
normally but there is no J-Space to show. This tool is for the case where a
harness is pointed at a **local open-weights model**.

## Quickstart (no GPU)

The mock backend has **zero dependencies** (stdlib only) and synthesises a
plausible bilingual read-out, so you can wire everything up first:

```bash
cd apps/serve
python -m j7scope_serve --backend mock
# open http://127.0.0.1:8799/  and, in another shell:
curl -N http://127.0.0.1:8799/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"stream":true,"messages":[{"role":"user","content":"hi"}]}'
```

The viewer lights up token by token. Everything from the mock backend is clearly
marked `DEMO`.

## Real read-out

Needs `torch` + `transformers` (from the repo root `pip install -e .`) and enough
VRAM for the model. The sidecar reuses [`j7scope.fitting.JLens`](../../j7scope/fitting.py):
`J_l` is fitted once on a small generic corpus (cached to disk), then every
generated token's residual is read out through it.

```bash
python -m j7scope_serve --backend hf \
  --model Qwen/Qwen2.5-7B-Instruct --layer 18 \
  --jacobian-cache ~/.cache/j7scope
```

## Record & replay (platform P1)

Live sessions are ephemeral; **traces** are the citable, deep-linkable artifact
(see [docs/platform-plan.md](../../docs/platform-plan.md)). Record every session
and serve them back for replay:

```bash
# record each session as a Trace v1, and serve traces for the replay viewer
python -m j7scope_serve --backend mock --record ../../results/traces

# or generate deterministic demo traces without a model (zero GPU):
python ../../experiments/build_demo_trace.py
python -m j7scope_serve --backend mock --traces ../../results/traces
```

Then open the replay viewer:

- `http://127.0.0.1:8799/?trace=demo-narrative-en` — play a trace token by token
- deep-link a moment: `…/?trace=demo-narrative-en#token=3`
- the trace picker (top-left) lists everything in `traces/index.json`

At each token the **rigor strip** shows the cross-lingual concept overlap against
a shuffled-pairing **null band** — a bar past the null band is a real signal, not
chance. The rigor layer (null, same-language ceiling, `sharedness` + CI) is
computed once by [`j7scope/rigor.py`](../../j7scope/rigor.py) and baked into the
trace at capture time; the viewer only displays it. Trace Schema v1 lives in
[`j7scope/trace.py`](../../j7scope/trace.py).

## Harness integration

- **opencode** → [`integrations/opencode/`](integrations/opencode/) (provider
  config + optional plugin).
- **codex / other OpenAI-compatible harnesses** → point the provider `base_url`
  at `http://127.0.0.1:8799/v1`; open the viewer for display. The side-channel is
  harness-agnostic, so any client works — only the convenience plugin is
  opencode-specific.

## HTTP surface

| route | method | purpose |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI chat, streaming or not; drives generation |
| `/v1/models` | GET | one model entry (harnesses probe this) |
| `/jspace/stream` | GET | SSE: one J-space event per generated token |
| `/` | GET | the standalone viewer |
| `/health` | GET | backend / model / layer / viewer count |

### Side-channel event

Each token emits (see [`protocol.py`](j7scope_serve/protocol.py)):

```jsonc
{
  "type": "token",
  "seq": 12,
  "request_id": "chatcmpl-…",   // correlates with the OpenAI response id
  "token": "deception",          // the surface token just emitted
  "layer": 18,
  "readout": {
    "zh": [{ "token": "欺骗", "score": 11.8, "rank": 0 }, …],
    "en": [{ "token": "deception", "score": 11.0, "rank": 1 }, …]
  }
}
```

The `zh` / `en` split is the live analogue of J7Scope's research question: a
*single* J-lens read-out, bucketed by script. When both columns light up on the
same concept at once, that workspace direction is shared across languages.

## Layout

```
apps/serve/
├── j7scope_serve/
│   ├── __main__.py     # CLI
│   ├── app.py          # stdlib HTTP server: OpenAI proxy + SSE side-channel
│   ├── backends.py     # MockBackend (stdlib) + HFBackend (torch, reuses JLens)
│   ├── bus.py          # thread-safe fan-out to viewers
│   └── protocol.py     # event schema + zh/en script bucketing
├── viewer/index.html   # self-contained live viewer
└── integrations/opencode/
```

> The mock backend is a wiring/demo aid, not an experiment. It never loads a
> model; its read-out is synthetic and marked `is_demo`. Research conclusions
> come only from the `hf` backend and the artifacts in [`../../results`](../../results).
