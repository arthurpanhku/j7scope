# J7Scope × opencode

Route [opencode](https://opencode.ai) through the J7Scope sidecar so a live
**J-Space read-out** appears next to your coding session — the concepts the
underlying open model is inclined to say next, split into 中文 / English columns.

opencode talks to the sidecar as an ordinary OpenAI-compatible provider; the
sidecar does the J-lens read-out and publishes it on a side-channel. **The
J-Space read-out only exists when opencode is pointed at a local open-weights
model through the sidecar** — it is not available for closed API models, whose
activations are never exposed.

## 1. Start the sidecar

Demo (no GPU, synthetic read-out — good for wiring it up first):

```bash
cd apps/serve
python -m j7scope_serve --backend mock
```

Real read-out (GPU box with the weights available):

```bash
pip install -e .            # from repo root: torch + transformers
cd apps/serve
python -m j7scope_serve --backend hf \
  --model Qwen/Qwen2.5-7B-Instruct --layer 18 \
  --jacobian-cache ~/.cache/j7scope
```

Open the viewer at <http://127.0.0.1:8799/>.

## 2. Point opencode at it

Copy [`opencode.json`](opencode.json) into your project root (or merge its
`provider` block into your existing config). Then select the model:

```bash
opencode
# /models  ->  J7Scope Sidecar  ->  j7scope-mock  (or the Qwen model)
```

or headless:

```bash
opencode run -m j7scope/j7scope-mock "say something about deception"
```

As opencode generates, the viewer at `/` lights up token by token.

## 3. (optional) Load the plugin

[`plugin/j7scope.ts`](plugin/j7scope.ts) is a small convenience layer: it
checks the sidecar is up on startup, logs the viewer URL, and can auto-open it.

```bash
mkdir -p .opencode/plugin
cp apps/serve/integrations/opencode/plugin/j7scope.ts .opencode/plugin/
# auto-open the viewer in your browser when opencode starts:
export J7SCOPE_OPEN=1
```

The plugin is optional — the read-out works from the provider config alone; the
plugin only surfaces the viewer link and an optional toast.

## Notes

- `J7SCOPE_SIDECAR` overrides the sidecar URL (default `http://127.0.0.1:8799`).
- The provider is a plain `@ai-sdk/openai-compatible` entry, so the same
  `opencode.json` pattern works for codex and any other OpenAI-compatible harness
  — only the display plugin is opencode-specific.
