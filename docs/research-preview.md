# J7Scope 0.2.0rc1 · Research Preview

J7Scope is a self-hosted observatory for LLM internal representations and reproducible
J-lens experiments. It combines live read-outs, activation patching, trace recording,
replay, and optional cross-language comparisons. Bilingual research is the first use
case, not a requirement for using or extending the tool.

This preview targets individual researchers and trusted lab groups. It is a single
model process with bounded FIFO admission, not a multi-tenant inference platform.
Open weights may run locally or on a cloud GPU. Ordinary commercial chat APIs cannot
supply the residual activations or differentiable computation needed by this backend.
A provider offering an explicit activation/intervention service can be integrated
through a backend plugin; no commercial-provider backend ships in this release.

## Install and start

Python 3.10+ is required. Download the wheel from the GitHub prerelease, then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install ./j7scope-0.2.0rc1-py3-none-any.whl
j7scope --backend mock --record ./traces
```

Open `http://127.0.0.1:8799/live.html`. From another terminal:

```bash
curl -N http://127.0.0.1:8799/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"stream":true,"messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

The mock is synthetic and labelled DEMO. The installed base package needs no
PyTorch; recording and metrics use NumPy. The source-only mock can still run with
stdlib alone if recording is disabled.

For real weights, install the HF extra from the downloaded wheel:

```bash
pip install './j7scope-0.2.0rc1-py3-none-any.whl[hf]'
j7scope --backend hf --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --model-revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --layer 1 --device cpu --dtype float32 --n-probes 8 \
  --jacobian-cache ./lenses --record ./traces --token-delay 0
```

For development, use `pip install -e '.[hf]'` and
`j7scope --config examples/smollm-cpu.json`. JSON config keys match CLI destinations
(e.g. `model_revision`, `max_new_tokens`); explicit flags override config defaults.
The CLI, viewer, and default concept vocabulary are included in wheels and sdists;
startup does not require a checkout or a special working directory.

## Validation and limits

```bash
j7scope-validate --output validation-real-model.json
```

This explicitly downloads a public pretrained 135M model (roughly 270 MB of weights).
It checks real generation, independent activation capture, lens finiteness, cache
round-trip, identity patching, trace schema, provenance, and authenticated HTTP JSON/SSE generation/recorded replay. The versioned local
CPU report is in [validation/smollm2-135m-cpu.json](validation/smollm2-135m-cpu.json).
A few random probes only verify the implementation path: their top tokens may be
semantically poor. They do not validate lens convergence or cross-language sharing.

| Path | Preview status |
|---|---|
| SmolLM2-135M, CPU float32 | Pretrained-model smoke validation included |
| Qwen2.5, dense HF structure | Reference configuration; GPU performance unvalidated in this release |
| Other dense HF decoders | Structural compatibility only; run validation for each model |
| GPU Docker profile | Configuration provided; needs NVIDIA runtime and hardware validation |
| Tool calling / agent execution | Unsupported; text-chat compatibility only |
| Commercial chat API | No activation read-out backend included |

The built-in service implements a text-only subset of `/v1/chat/completions`, not
the entire provider protocol. It rejects tool calls, non-text content, and structured
output requests. Usage currently counts emitted tokens only; prompt-token usage is
not measured. Existing harness configuration examples demonstrate wiring, not a
complete agent tool loop.

For research-grade fitting, use `experiments/fit_paper_jacobian.py` and pass the
resulting `.pt` plus matching `.json` via `--jacobian-file`. The server checks model,
revision, layer, estimator, tensor hash, and shape. See the
[GPU runbook](gpu-cloud-runbook.md). All newly recorded traces remain `preview:true`;
promoting them to evidence requires an independently reviewed experiment protocol.

## Remote operation and session isolation

Keep the service on loopback and use SSH forwarding for a private cloud machine:

```bash
# On the GPU machine (after installing the HF extra):
export J7SCOPE_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
j7scope --backend hf --model Qwen/Qwen2.5-7B-Instruct --layer 18 \
  --record ./traces --jacobian-cache ./lenses --token-delay 0

# On the researcher's laptop:
ssh -N -L 8799:127.0.0.1:8799 user@your-gpu-host
```

For direct remote binding, add `--host 0.0.0.0`. A 32-character ASCII key in
`J7SCOPE_API_KEY` is mandatory on non-loopback binds; keys are never CLI flags.
For public-network access, place the service behind an HTTPS reverse proxy, turn
off proxy buffering for SSE, and add `--secure-cookie`. Do not terminate TLS in
this stdlib research server. Origins are same-origin by default; explicitly allow
additional origins with repeated `--allowed-origin https://your-viewer.example`.
The unauthenticated `/health` endpoint reveals only liveness, not model details.
Authenticated `/ready` reports model readiness, capabilities, and queue depth.

The operator creates one separate workspace for each researcher:

```bash
curl -X POST http://127.0.0.1:8799/jspace/sessions \
  -H "Authorization: Bearer $J7SCOPE_API_KEY"
# {"session_id":"...", "api_key":"...", "expires_in":86400}
```

Give that session key to its researcher. They use it as their client's Bearer API
key, and paste the same key into `/connect.html` to connect the viewer. The browser
receives an HttpOnly, SameSite=Strict cookie; keys never enter URLs or localStorage.
The operator's direct requests use a separate operator workspace. There is no
cross-session event broadcast, backlog replay, gallery, or trace access. Session
credentials cannot create other sessions. Browser logout revokes that browser
cookie; session keys expire after `--session-ttl` (24 hours by default). Restarting
the process revokes all ephemeral credentials; recordings remain on disk, and the
operator can recover/export them from the filesystem. There is no account database.

With auth enabled, recordings are stored as `<record-dir>/<session-id>/<trace-id>`;
the operator workspace uses `operator`. With auth disabled on loopback, the existing
flat trace layout is retained. Export only a chosen session's trace folder when
sharing offline artifacts. Run one service process per data directory/model.

The model runs one task at a time so generation and patching hooks cannot race.
Defaults: four queued jobs, 30-second queue wait, 512 generated tokens, 256 KiB
request bodies, 128 messages, 16 SSE viewers, 64 connections. A full/timed-out queue
returns HTTP 503 with `Retry-After`. Generation time limits are cooperative between
forward passes (`--request-timeout`); they cannot interrupt a hung GPU kernel or
an intervention mid-forward. Put process supervision and proxy timeouts around the
service. Slow viewers drop events rather than blocking computation; SSE is a live
view, while complete recorded traces are the durable result. Failed/disconnected
requests do not become complete traces. Trace publication and index replacement
are atomic; recording failures appear in operator logs.

## Containers

```bash
export J7SCOPE_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
docker compose up --build demo
# NVIDIA GPU profile, exposed on local port 8800:
docker compose --profile gpu up --build gpu
```

Containers run as a non-root user, keep data in named volumes, and expose host ports
on loopback. Demo and GPU profiles use separate volumes. The GPU profile uses a
preview-fitted lens; mount a reviewed lens and use `--jacobian-file` for research.
Docker daemon/GPU runtime availability must be verified on the deployment machine.

## Extend models, backends, and research concepts

See [extending.md](extending.md) for plugin contracts. `j7scope --list-backends`
shows installed backends. `--corpus path.json` selects a nonempty JSON array of
preview-fitting prompts; `--lexicon path.json` loads a token→concept map for recorded
metrics. The packaged bilingual vocabulary is just a starting point: concept
coverage and tokenization must be reviewed for each model/language. A trace records
its vocabulary hash and entry count; raw activations and weights are not exported.
