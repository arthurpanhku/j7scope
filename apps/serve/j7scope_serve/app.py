"""Stdlib HTTP server: an OpenAI-compatible proxy with a J-space side-channel.

Routes
------
GET  /                     the standalone J-Space viewer page
GET  /health               liveness + current config
GET  /v1/models            one model entry (harnesses probe this)
POST /v1/chat/completions  OpenAI streaming chat; drives generation
GET  /jspace/stream        Server-Sent Events: one J-space event per token

The chat handler runs the backend, streams tokens back to the caller in OpenAI
format, and *simultaneously* publishes each token's bucketed J-lens readout onto
the bus, which fans it out to every connected viewer. No dependency beyond the
standard library, so it runs anywhere the mock backend does.
"""

from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from . import protocol
from .backends import Backend
from .bus import ReadoutBus

SITE_DIR = Path(__file__).resolve().parents[2] / "site"   # apps/site (gallery/replay/…)


class SidecarServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, backend: Backend, *, per_lang: int = 8,
                 token_delay: float = 0.04, record_dir=None, repo_root=None,
                 traces_dir=None):
        super().__init__(addr, _Handler)
        self.backend = backend
        self.bus = ReadoutBus()
        self.per_lang = per_lang
        self.token_delay = token_delay
        self.record_dir = Path(record_dir) if record_dir else None
        # Replay viewer reads traces from here; defaults to the record dir.
        self.traces_dir = Path(traces_dir) if traces_dir else self.record_dir
        self.repo_root = Path(repo_root) if repo_root else None
        self._lexicon = None  # built lazily on first recorded session

    def lexicon(self):
        if self._lexicon is None:
            from .recorder import build_lexicon
            self._lexicon = build_lexicon(self.repo_root, self.backend)
        return self._lexicon


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # Quieter logging: one line per request without the default noise.
    def log_message(self, fmt, *args):  # noqa: A003
        pass

    # ---- helpers ----------------------------------------------------------

    @property
    def bus(self) -> ReadoutBus:
        return self.server.bus  # type: ignore[attr-defined]

    @property
    def backend(self) -> Backend:
        return self.server.backend  # type: ignore[attr-defined]

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    def _begin_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self._cors()
        self.end_headers()

    def _sse_send(self, obj) -> None:
        self.wfile.write(b"data: ")
        self.wfile.write(json.dumps(obj, ensure_ascii=False).encode("utf-8"))
        self.wfile.write(b"\n\n")
        self.wfile.flush()

    # ---- routing ----------------------------------------------------------

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/health":
            return self._json({
                "ok": True,
                "backend": type(self.backend).__name__,
                "model": self.backend.model_name,
                "layer": self.backend.layer,
                "is_demo": getattr(self.backend, "is_demo", False),
                "viewers": self.bus.subscriber_count(),
            })
        if path == "/v1/models":
            return self._json({
                "object": "list",
                "data": [{
                    "id": self.backend.model_name,
                    "object": "model",
                    "owned_by": "j7scope",
                }],
            })
        if path == "/jspace/stream":
            return self._stream_jspace()
        if path == "/traces/index.json" or path.startswith("/traces/"):
            return self._serve_trace_file(path[len("/traces/"):])
        # everything else: static file from the site dir (gallery / replay / …)
        rel = "index.html" if path == "/" else path.lstrip("/")
        return self._serve_static(SITE_DIR, rel)

    def do_POST(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/v1/chat/completions":
            return self._chat_completions()
        return self._json({"error": "not found"}, status=404)

    # ---- handlers ---------------------------------------------------------

    _CONTENT_TYPES = {".json": "application/json; charset=utf-8",
                      ".jsonl": "application/x-ndjson; charset=utf-8",
                      ".html": "text/html; charset=utf-8",
                      ".css": "text/css; charset=utf-8",
                      ".js": "application/javascript; charset=utf-8",
                      ".svg": "image/svg+xml"}

    def _serve_from(self, base_dir, rel: str) -> None:
        """Serve rel from base_dir with a path-traversal guard."""
        if not base_dir:
            return self._json({"error": "no dir configured"}, status=404)
        base = base_dir.resolve()
        target = (base / rel).resolve()
        if base != target and base not in target.parents:  # path-traversal guard
            return self._json({"error": "forbidden"}, status=403)
        if not target.is_file():
            return self._json({"error": "not found"}, status=404)
        body = target.read_bytes()
        ctype = self._CONTENT_TYPES.get(target.suffix, "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _serve_trace_file(self, rel: str) -> None:
        return self._serve_from(self.server.traces_dir, rel)  # type: ignore[attr-defined]

    def _serve_static(self, base_dir, rel: str) -> None:
        return self._serve_from(base_dir, rel)

    def _stream_jspace(self) -> None:
        self._begin_sse()
        sub_id, q = self.bus.subscribe(replay=True)
        self._sse_send(protocol.meta_event(
            model=self.backend.model_name,
            layer=self.backend.layer,
            backend=type(self.backend).__name__,
            is_demo=getattr(self.backend, "is_demo", False),
        ))
        try:
            while True:
                try:
                    event = q.get(timeout=15.0)
                    self._sse_send(event)
                except Exception:  # queue.Empty -> heartbeat comment
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.bus.unsubscribe(sub_id)

    def _chat_completions(self) -> None:
        body = self._read_body()
        messages = body.get("messages") or []
        model = body.get("model") or self.backend.model_name
        stream = bool(body.get("stream", False))
        params = {
            "max_tokens": body.get("max_tokens"),
            "temperature": body.get("temperature", 0.0),
        }

        request_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        per_lang = self.server.per_lang          # type: ignore[attr-defined]
        token_delay = self.server.token_delay    # type: ignore[attr-defined]

        self.bus.publish(protocol.request_start_event(request_id=request_id, model=model))

        if stream:
            self._begin_sse()
            self._sse_send(protocol.chat_chunk(
                request_id=request_id, model=model, created=created, role="assistant"))

        full_text = []
        buffered = []          # per-token records for --record
        seq = 0
        t_start = time.time()
        try:
            for step in self.backend.generate(messages, **params):
                full_text.append(step.text)
                readout = protocol.bucket_readout(step.topk, per_lang=per_lang)
                self.bus.publish(protocol.token_event(
                    seq=seq, request_id=request_id, token=step.token,
                    readout=readout, model=self.backend.model_name,
                    layer=self.backend.layer))
                if self.server.record_dir:  # type: ignore[attr-defined]
                    buffered.append({
                        "seq": seq,
                        "ts_rel": round(time.time() - t_start, 3),
                        "token": step.token,
                        "token_script": protocol.script_of(step.token),
                        "readout": readout,
                    })
                seq += 1
                if stream:
                    self._sse_send(protocol.chat_chunk(
                        request_id=request_id, model=model, created=created,
                        content=step.text))
                if token_delay:
                    time.sleep(token_delay)
        except (BrokenPipeError, ConnectionResetError):
            self.bus.publish(protocol.request_end_event(request_id=request_id, n_tokens=seq))
            self._maybe_record(buffered, messages)
            return

        self.bus.publish(protocol.request_end_event(request_id=request_id, n_tokens=seq))
        self._maybe_record(buffered, messages)

        if stream:
            self._sse_send(protocol.chat_chunk(
                request_id=request_id, model=model, created=created,
                finish_reason="stop"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            text = "".join(full_text)
            self._json({
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": seq,
                          "total_tokens": seq},
            })


    def _maybe_record(self, buffered, messages) -> None:
        record_dir = self.server.record_dir  # type: ignore[attr-defined]
        if not record_dir or not buffered:
            return
        prompt = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                prompt = m.get("content", "")
                break
        try:
            from .recorder import record_trace
            path = record_trace(record_dir, backend=self.backend, prompt=prompt,
                                buffered=buffered, lexicon=self.server.lexicon())  # type: ignore[attr-defined]
            if path:
                print(f"  recorded trace -> {path}")
        except Exception as exc:  # recording must never break the response
            print(f"  [record] failed: {exc!r}")


def serve(backend: Backend, *, host: str = "127.0.0.1", port: int = 8799,
          per_lang: int = 8, token_delay: float = 0.04,
          record_dir=None, repo_root=None, traces_dir=None) -> None:
    server = SidecarServer((host, port), backend, per_lang=per_lang,
                           token_delay=token_delay, record_dir=record_dir,
                           repo_root=repo_root, traces_dir=traces_dir)
    demo = " [DEMO]" if getattr(backend, "is_demo", False) else ""
    print(f"j7scope-serve{demo}  backend={type(backend).__name__} "
          f"model={backend.model_name} layer={backend.layer}")
    print(f"  OpenAI endpoint : http://{host}:{port}/v1")
    print(f"  Gallery         : http://{host}:{port}/")
    print(f"  Live view       : http://{host}:{port}/live.html")
    print(f"  J-Space stream  : http://{host}:{port}/jspace/stream")
    if record_dir:
        print(f"  recording traces: {record_dir}")
    if server.traces_dir:
        print(f"  serving traces  : {server.traces_dir}  (replay: /?trace=<id>)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.shutdown()
