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
import logging
import math
import threading
from http.cookies import SimpleCookie
from urllib.parse import urlsplit
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from . import protocol
from .backends import Backend
from .bus import ReadoutBus
from .runtime import TaskGate, BusyError
from .security import Credentials

_LOG = logging.getLogger(__name__)


class RequestError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status

SITE_DIR = (Path(__file__).parent / "static" if (Path(__file__).parent / "static").is_dir()
            else Path(__file__).resolve().parents[2] / "site")   # apps/site (gallery/replay/…)


class SidecarServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, backend: Backend, *, per_lang: int = 8,
                 token_delay: float = 0.04, record_dir=None, repo_root=None,
                 traces_dir=None, api_key=None, allowed_origins=(), max_queue=4,
                 queue_timeout=30, request_timeout=120, max_body=262144,
                 max_tokens=512, max_viewers=16, secure_cookie=False,
                 session_ttl=86400, max_connections=64, lexicon_file=None):
        if addr[0] not in ("127.0.0.1", "localhost", "::1") and not api_key:
            raise ValueError("Remote binding requires J7SCOPE_API_KEY (at least 32 characters)")
        if api_key and (len(api_key) < 32 or not api_key.isascii()):
            raise ValueError("API key must contain at least 32 ASCII characters")
        super().__init__(addr, _Handler)
        self.credentials = Credentials(api_key, ttl=session_ttl)
        self.allowed_origins = set(allowed_origins)
        self.gate = TaskGate(max_queue, queue_timeout)
        self.request_timeout = request_timeout
        self.max_body = max_body
        self.max_tokens = max_tokens
        self.max_viewers = max_viewers
        self.viewer_slots = threading.BoundedSemaphore(max_viewers)
        self.connections = threading.BoundedSemaphore(max_connections)
        self.secure_cookie = secure_cookie
        self.backend = backend
        self.bus = ReadoutBus()
        self.per_lang = per_lang
        self.token_delay = token_delay
        self.record_dir = Path(record_dir) if record_dir else None
        # Replay viewer reads traces from here; defaults to the record dir.
        self.traces_dir = Path(traces_dir) if traces_dir else self.record_dir
        self.repo_root = Path(repo_root) if repo_root else None
        self.lexicon_file = lexicon_file
        self._lexicon = None  # built lazily on first recorded session

    def process_request(self, request, client_address):
        if not self.connections.acquire(blocking=False):
            try:
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            finally:
                self.shutdown_request(request)
            return
        request.settimeout(30)
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()

    def lexicon(self):
        if self._lexicon is None:
            from .recorder import build_lexicon
            lexicon = build_lexicon(self.repo_root, self.backend)
            if self.lexicon_file:
                extra = json.loads(Path(self.lexicon_file).read_text(encoding="utf-8"))
                if not isinstance(extra, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in extra.items()):
                    raise ValueError("lexicon must map surface tokens to concept strings")
                lexicon.update(extra)
            self._lexicon = lexicon
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
        origin = self.headers.get("Origin")
        if origin in self.server.allowed_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Credentials", "true")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")

    def _json(self, obj, status: int = 200, *, headers=None) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _token(self):
        auth = self.headers.get("Authorization", "")
        if auth:
            return auth[7:] if auth.startswith("Bearer ") else ""
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
            return cookie["j7scope_session"].value if "j7scope_session" in cookie else ""
        except Exception:
            return ""

    def _authorize(self):
        # Cookies never grant cross-origin access, even when a client ignores CORS.
        origin = self.headers.get("Origin")
        same_origin = origin and urlsplit(origin).netloc == self.headers.get("Host")
        if origin and not same_origin and origin not in self.server.allowed_origins:
            self.close_connection = True
            self._json({"error": "origin not allowed"}, 403)
            return False
        self.principal = self.server.credentials.resolve(self._token())
        if self.principal is None:
            self.close_connection = True
            self._json({"error": "authentication required; open /connect.html"}, 401,
                       headers={"WWW-Authenticate": "Bearer"})
            return False
        return True

    def _read_body(self) -> dict:
        if len(self.headers.get_all("Content-Length", [])) > 1:
            self.close_connection = True
            raise RequestError("duplicate Content-Length")
        if self.headers.get("Transfer-Encoding"):
            self.close_connection = True
            raise RequestError("chunked request bodies are unsupported")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise RequestError("invalid Content-Length")
        if length < 0 or length > self.server.max_body:
            self.close_connection = True
            raise RequestError("request body exceeds limit", 413)
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise RequestError("incomplete body")
        try:
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise RequestError("expected a JSON object")
        if not isinstance(body, dict):
            raise RequestError("expected a JSON object")
        return body

    def _scope_dir(self, base):
        if not base:
            return None
        return Path(base) / self.principal.session if self.server.credentials.api_key else Path(base)

    def _begin_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.close_connection = True
        self.send_header("Connection", "close")
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
        path = urlsplit(self.path).path
        if path == "/health":
            return self._json({"ok": True, "service": "j7scope"})
        protected = path in ("/ready", "/v1/models", "/jspace/stream", "/jspace/session") or path.startswith("/traces/")
        if protected and not self._authorize():
            return
        if path == "/ready":
            ready = self.backend.ready()
            return self._json({"ready": ready, "model": self.backend.model_name,
                "backend": type(self.backend).__name__, "layer": self.backend.layer,
                "is_demo": getattr(self.backend, "is_demo", False),
                "queue_depth": self.server.gate.waiting,
                "capabilities": self.backend.capabilities()}, 200 if ready else 503)
        if path == "/jspace/session":
            return self._json({"session_id": self.principal.session, "operator": self.principal.operator})
        if path == "/v1/models":
            return self._json({"object": "list", "data": [{"id": self.backend.model_name,
                               "object": "model", "owned_by": "j7scope"}]})
        if path == "/jspace/stream":
            if not self.server.viewer_slots.acquire(blocking=False):
                return self._json({"error": "viewer limit reached"}, 503)
            try:
                return self._stream_jspace()
            finally:
                self.server.viewer_slots.release()
        if path.startswith("/traces/"):
            return self._serve_from(self._scope_dir(self.server.traces_dir), path[len("/traces/"):])
        rel = "index.html" if path == "/" else path.lstrip("/")
        return self._serve_static(SITE_DIR, rel)

    def do_POST(self):  # noqa: N802
        if not self._authorize():
            return
        path = urlsplit(self.path).path
        try:
            if path in ("/auth/login", "/auth/logout", "/jspace/sessions"):
                if self.headers.get("Content-Length", "0") != "0":
                    self._read_body()
            if path == "/auth/login":
                token, principal = self.server.credentials.issue(self.principal)
                cookie = f"j7scope_session={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age={self.server.credentials.ttl}"
                if self.server.secure_cookie:
                    cookie += "; Secure"
                return self._json({"session_id": principal.session}, headers={"Set-Cookie": cookie})
            if path == "/auth/logout":
                self.server.credentials.revoke(self._token())
                return self._json({"ok": True}, headers={"Set-Cookie": "j7scope_session=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0"})
            if path == "/jspace/sessions":
                if not self.principal.operator:
                    return self._json({"error": "operator credential required"}, 403)
                token, principal = self.server.credentials.issue()
                return self._json({"session_id": principal.session, "api_key": token,
                                   "expires_in": self.server.credentials.ttl}, 201)
            if path == "/v1/chat/completions":
                return self._chat_completions()
            if path == "/jspace/interventions":
                with self.server.gate.enter():
                    return self._run_intervention()
            return self._json({"error": "not found"}, 404)
        except RequestError as exc:
            self.close_connection = True
            return self._json({"error": str(exc)}, exc.status)
        except BusyError as exc:
            self.close_connection = True
            return self._json({"error": str(exc)}, 503, headers={"Retry-After": "5"})
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True
        except Exception:
            _LOG.exception("request failed")
            self.close_connection = True
            return self._json({"error": "request failed; inspect server logs"}, 500)

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
        sub_id, q = self.bus.subscribe(replay=True, session=self.principal.session)
        self._sse_send(protocol.meta_event(
            model=self.backend.model_name,
            layer=self.backend.layer,
            backend=type(self.backend).__name__,
            is_demo=getattr(self.backend, "is_demo", False),
        ))
        try:
            deadline = time.monotonic() + self.server.credentials.ttl
            while time.monotonic() < deadline and self.server.credentials.resolve(self._token()) is not None:
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
        messages = body.get("messages")
        if not isinstance(messages, list) or not 1 <= len(messages) <= 128:
            raise RequestError("messages must contain between 1 and 128 text messages")
        if any(not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant")
               or not isinstance(m.get("content"), str) for m in messages):
            raise RequestError("only system/user/assistant text messages are supported")
        if body.get("tools") or body.get("tool_choice") or body.get("response_format"):
            raise RequestError("tools and structured output are not supported in this preview")
        if body.get("model", self.backend.model_name) != self.backend.model_name:
            raise RequestError("requested model is not loaded", 404)
        max_tokens = body.get("max_tokens", self.server.max_tokens)
        temperature = body.get("temperature", 0.0)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= self.server.max_tokens:
            raise RequestError(f"max_tokens must be between 1 and {self.server.max_tokens}")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise RequestError("temperature must be between 0 and 2")
        if "stream" in body and not isinstance(body["stream"], bool):
            raise RequestError("stream must be a boolean")
        with self.server.gate.enter():
            return self._generate_chat(messages, bool(body.get("stream", False)),
                                       {"max_tokens": max_tokens, "temperature": temperature})

    def _generate_chat(self, messages, stream, params):
        request_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        model = self.backend.model_name
        scope = self.principal.session
        publish = lambda event: self.bus.publish(event, session=scope)
        full_text, buffered = [], []
        seq = 0
        t_start = time.monotonic()
        finish_reason = "stop"
        completed = False
        iterator = None
        publish(protocol.request_start_event(request_id=request_id, model=model))
        if stream:
            self._begin_sse()
            self._sse_send(protocol.chat_chunk(request_id=request_id, model=model, created=created, role="assistant"))
        try:
            iterator = iter(self.backend.generate(messages, **params))
            for step in iterator:
                if time.monotonic() - t_start > self.server.request_timeout:
                    raise TimeoutError("generation exceeded time limit")
                full_text.append(step.text)
                readout = protocol.bucket_readout(step.topk, per_lang=self.server.per_lang)
                publish(protocol.token_event(seq=seq, request_id=request_id, token=step.token,
                    readout=readout, model=model, layer=self.backend.layer))
                if self.server.record_dir:
                    buffered.append({"seq": seq, "ts_rel": round(time.monotonic() - t_start, 3),
                        "token": step.token, "token_script": protocol.script_of(step.token), "readout": readout})
                seq += 1
                if stream:
                    self._sse_send(protocol.chat_chunk(request_id=request_id, model=model, created=created, content=step.text))
                if seq >= params["max_tokens"]:
                    finish_reason = "length"
                    break
                if self.server.token_delay:
                    time.sleep(self.server.token_delay)
            completed = True
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
            return
        except Exception:
            _LOG.exception("generation failed request_id=%s", request_id)
            if stream:
                self._sse_send({"error": {"message": "generation failed or exceeded time limit", "type": "server_error"}})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            return self._json({"error": "generation failed or exceeded time limit"}, 500)
        finally:
            if iterator is not None and hasattr(iterator, "close"):
                iterator.close()
            event = protocol.request_end_event(request_id=request_id, n_tokens=seq)
            event["status"] = "complete" if completed else "interrupted"
            publish(event)
            _LOG.info("generation request_id=%s status=%s tokens=%s", request_id, event["status"], seq)
            # Interrupted generations must not appear as complete research artifacts.
            if completed:
                self._maybe_record(buffered, messages)
        if stream:
            self._sse_send(protocol.chat_chunk(request_id=request_id, model=model, created=created, finish_reason=finish_reason))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            self._json({"id": request_id, "object": "chat.completion", "created": created,
                "model": model, "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(full_text)},
                "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": 0, "completion_tokens": seq, "total_tokens": seq}})

    def _run_intervention(self) -> None:
        if not getattr(self.backend, "supports_intervention", False):
            return self._json({
                "error": "activation patching requires the hf backend on a GPU sidecar",
                "capability": "activation_patch",
            }, status=409)
        body = self._read_body()
        source_prompt = body.get("source_prompt")
        target_prompt = body.get("target_prompt")
        if not isinstance(source_prompt, str) or not source_prompt.strip():
            return self._json({"error": "source_prompt is required"}, status=400)
        if not isinstance(target_prompt, str) or not target_prompt.strip():
            return self._json({"error": "target_prompt is required"}, status=400)
        if len(source_prompt) > 20_000 or len(target_prompt) > 20_000:
            return self._json({"error": "prompt exceeds 20,000 characters"}, status=400)
        try:
            patch_layer = int(body.get("patch_layer"))
            source_position = int(body.get("source_position", -1))
            target_position = int(body.get("target_position", -1))
            topk = min(100, max(1, int(body.get("topk", 20))))
        except (TypeError, ValueError):
            return self._json({"error": "intervention positions and layers must be integers"}, status=400)
        if patch_layer < 0 or patch_layer >= self.backend.layer:
            return self._json({
                "error": f"patch_layer must be between 0 and {self.backend.layer - 1}"
            }, status=400)
        try:
            result = self.backend.intervene(
                source_prompt=source_prompt,
                target_prompt=target_prompt,
                patch_layer=patch_layer,
                source_position=source_position,
                target_position=target_position,
                topk=topk,
            )
        except (ValueError, IndexError) as exc:
            return self._json({"error": str(exc)}, status=400)
        except Exception as exc:
            print(f"  [intervention] failed: {exc!r}")
            return self._json({
                "error": "activation patching failed; inspect the sidecar log"
            }, status=500)
        raw_readout = result.pop("readout")
        result["readout"] = protocol.bucket_readout(
            raw_readout,
            per_lang=self.server.per_lang,  # type: ignore[attr-defined]
        )
        result["is_demo"] = False
        return self._json(result)

    def _maybe_record(self, buffered, messages) -> None:
        record_dir = self._scope_dir(self.server.record_dir)
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
          record_dir=None, repo_root=None, traces_dir=None, **server_options) -> None:
    server = SidecarServer((host, port), backend, per_lang=per_lang,
                           token_delay=token_delay, record_dir=record_dir,
                           repo_root=repo_root, traces_dir=traces_dir, **server_options)
    if server.record_dir or server.lexicon_file:
        server.lexicon()
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
        pass
    finally:
        server.server_close()
