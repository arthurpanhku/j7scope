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

VIEWER_HTML = Path(__file__).resolve().parents[1] / "viewer" / "index.html"


class SidecarServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, backend: Backend, *, per_lang: int = 8,
                 token_delay: float = 0.04):
        super().__init__(addr, _Handler)
        self.backend = backend
        self.bus = ReadoutBus()
        self.per_lang = per_lang
        self.token_delay = token_delay


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
        if path in ("/", "/index.html"):
            return self._serve_viewer()
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
        return self._json({"error": "not found"}, status=404)

    def do_POST(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/v1/chat/completions":
            return self._chat_completions()
        return self._json({"error": "not found"}, status=404)

    # ---- handlers ---------------------------------------------------------

    def _serve_viewer(self) -> None:
        try:
            body = VIEWER_HTML.read_bytes()
        except OSError:
            return self._json({"error": "viewer not found"}, status=500)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

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
        seq = 0
        try:
            for step in self.backend.generate(messages, **params):
                full_text.append(step.text)
                readout = protocol.bucket_readout(step.topk, per_lang=per_lang)
                self.bus.publish(protocol.token_event(
                    seq=seq, request_id=request_id, token=step.token,
                    readout=readout, model=self.backend.model_name,
                    layer=self.backend.layer))
                seq += 1
                if stream:
                    self._sse_send(protocol.chat_chunk(
                        request_id=request_id, model=model, created=created,
                        content=step.text))
                if token_delay:
                    time.sleep(token_delay)
        except (BrokenPipeError, ConnectionResetError):
            self.bus.publish(protocol.request_end_event(request_id=request_id, n_tokens=seq))
            return

        self.bus.publish(protocol.request_end_event(request_id=request_id, n_tokens=seq))

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


def serve(backend: Backend, *, host: str = "127.0.0.1", port: int = 8799,
          per_lang: int = 8, token_delay: float = 0.04) -> None:
    server = SidecarServer((host, port), backend, per_lang=per_lang,
                           token_delay=token_delay)
    demo = " [DEMO]" if getattr(backend, "is_demo", False) else ""
    print(f"j7scope-serve{demo}  backend={type(backend).__name__} "
          f"model={backend.model_name} layer={backend.layer}")
    print(f"  OpenAI endpoint : http://{host}:{port}/v1")
    print(f"  J-Space viewer  : http://{host}:{port}/")
    print(f"  J-Space stream  : http://{host}:{port}/jspace/stream")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.shutdown()
