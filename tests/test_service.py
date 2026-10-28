"""HTTP-level reliability, privacy, and admission tests for the preview server."""
import http.client
import json
from pathlib import Path
import sys
import threading
import time
from contextlib import contextmanager

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps/serve"))
from j7scope_serve.app import SidecarServer
from j7scope_serve.backends import MockBackend, Step
from j7scope_serve.bus import ReadoutBus
from j7scope_serve.runtime import TaskGate, BusyError
from j7scope_serve.security import Credentials

KEY = "test-operator-key-" + "a" * 32


@contextmanager
def running(backend=None, **options):
    server = SidecarServer(("127.0.0.1", 0), backend or MockBackend(), token_delay=0, **options)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def request(server, method, path, body=None, token=None, headers=None):
    connection = http.client.HTTPConnection(*server.server_address, timeout=5)
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    if token:
        hdrs["Authorization"] = "Bearer " + token
    connection.request(method, path, body=json.dumps(body) if body is not None else None, headers=hdrs)
    response = connection.getresponse()
    raw = response.read()
    result = (response.status, dict(response.getheaders()), raw)
    connection.close()
    return result


def chat(**kw):
    return {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 2, **kw}


def test_remote_bind_requires_strong_key():
    with pytest.raises(ValueError, match="Remote binding"):
        SidecarServer(("0.0.0.0", 0), MockBackend())
    with pytest.raises(ValueError, match="32 ASCII"):
        SidecarServer(("127.0.0.1", 0), MockBackend(), api_key="short")


def test_authentication_and_cross_origin_denial():
    with running(api_key=KEY) as server:
        assert request(server, "GET", "/health")[0] == 200
        for path in ("/ready", "/v1/models", "/jspace/stream", "/traces/index.json"):
            assert request(server, "GET", path)[0] == 401
        assert request(server, "GET", "/ready", token=KEY)[0] == 200
        assert request(server, "GET", "/ready", token=KEY,
                       headers={"Origin": "https://attacker.example"})[0] == 403
        response = request(server, "GET", "/ready", token=KEY)
        assert "Access-Control-Allow-Origin" not in response[1]
        assert b"operator-key" not in response[2]


def test_browser_cookie_login_logout_and_session_permissions():
    with running(api_key=KEY, secure_cookie=True) as server:
        code, headers, raw = request(server, "POST", "/auth/login", token=KEY)
        assert code == 200
        cookie = headers["Set-Cookie"]
        assert all(word in cookie for word in ("HttpOnly", "SameSite=Strict", "Secure"))
        assert KEY not in cookie
        cookie_header = {"Cookie": cookie.split(";", 1)[0]}
        assert request(server, "GET", "/ready", headers=cookie_header)[0] == 200
        assert request(server, "POST", "/auth/logout", headers=cookie_header)[0] == 200
        assert request(server, "GET", "/ready", headers=cookie_header)[0] == 401
        session = json.loads(request(server, "POST", "/jspace/sessions", token=KEY)[2])
        assert request(server, "POST", "/jspace/sessions", token=session["api_key"])[0] == 403
        assert request(server, "GET", "/jspace/session", token=session["api_key"])[0] == 200


def test_session_event_and_recording_isolation(tmp_path):
    with running(api_key=KEY, record_dir=tmp_path) as server:
        a, b = [json.loads(request(server, "POST", "/jspace/sessions", token=KEY)[2]) for _ in range(2)]
        _, qa = server.bus.subscribe(session=a["session_id"])
        _, qb = server.bus.subscribe(session=b["session_id"])
        assert request(server, "POST", "/v1/chat/completions", chat(), a["api_key"])[0] == 200
        assert not qa.empty() and qb.empty()
        assert request(server, "GET", "/traces/index.json", token=a["api_key"])[0] == 200
        assert request(server, "GET", "/traces/index.json", token=b["api_key"])[0] == 404
        assert request(server, "GET", "/traces/../" + a["session_id"] + "/index.json", token=b["api_key"])[0] == 403
        # Reconnection backlog is scoped too.
        _, replay_b = server.bus.subscribe(session=b["session_id"])
        assert replay_b.empty()


@pytest.mark.parametrize("body", [[], {}, chat(messages=[]), chat(max_tokens=0), chat(max_tokens=True),
    chat(max_tokens=100000), chat(temperature=float("nan")), chat(temperature=-1),
    chat(stream="true"), chat(tools=[{}]), chat(messages=[{"role":"user", "content":[]}])])
def test_invalid_chat_does_not_enter_model(body):
    with running() as server:
        assert request(server, "POST", "/v1/chat/completions", body)[0] == 400


def test_body_limit_and_model_validation():
    with running(max_body=100) as server:
        assert request(server, "POST", "/v1/chat/completions", {"huge":"a"*200})[0] == 413
    with running() as server:
        assert request(server, "POST", "/v1/chat/completions", chat(model="unknown"))[0] == 404
        status, _, raw = request(server, "POST", "/v1/chat/completions", chat(stream=True))
        assert status == 200 and raw.endswith(b"data: [DONE]\n\n")
        assert b'"finish_reason": "length"' in raw


def test_model_serialization_and_bounded_queue():
    entered, release = threading.Event(), threading.Event()

    class Blocking(MockBackend):
        def generate(self, messages, **kw):
            entered.set()
            release.wait(3)
            yield Step("ok", "ok", [])

    with running(Blocking(), max_queue=0) as server:
        first = threading.Thread(target=lambda: request(server, "POST", "/v1/chat/completions", chat()))
        first.start()
        assert entered.wait(2)
        code, headers, _ = request(server, "POST", "/v1/chat/completions", chat())
        assert code == 503 and headers["Retry-After"] == "5"
        release.set()
        first.join(3)
        assert request(server, "POST", "/v1/chat/completions", chat())[0] == 200


def test_failures_release_gate_and_do_not_record_partial_traces(tmp_path):
    class Failing(MockBackend):
        def generate(self, messages, **kw):
            yield Step("partial", "partial", [])
            raise RuntimeError("private provider error")
    with running(Failing(), record_dir=tmp_path) as server:
        for stream in (False, True):
            code, _, raw = request(server, "POST", "/v1/chat/completions", chat(stream=stream, max_tokens=4))
            assert code == (200 if stream else 500)
            assert b"private provider error" not in raw
            assert b"error" in raw
        assert not list(tmp_path.glob("rec-*"))


def test_gate_timeout_and_credential_expiry(monkeypatch):
    gate = TaskGate(max_queue=1, wait_timeout=0.01)
    with gate.enter():
        with pytest.raises(BusyError, match="timed out"):
            with gate.enter():
                pass
    with gate.enter():
        pass
    credentials = Credentials(KEY, ttl=1)
    token, _ = credentials.issue()
    assert credentials.resolve(token)
    monkeypatch.setattr("j7scope_serve.security.time.monotonic", lambda: 1e30)
    assert credentials.resolve(token) is None


def test_slow_subscriber_does_not_block_generation():
    bus = ReadoutBus(queue_max=1)
    _, q = bus.subscribe()
    for i in range(100):
        bus.publish({"seq": i})
    assert q.qsize() == 1


def test_generation_and_intervention_share_admission():
    entered, release = threading.Event(), threading.Event()
    class Instrumented(MockBackend):
        supports_intervention = True
        def generate(self, messages, **kw):
            entered.set()
            release.wait(3)
            yield Step("ok", "ok", [])
        def intervene(self, **kw):
            assert release.is_set()
            return {"readout": [("concept", 1.0)], "next_token": []}
    with running(Instrumented(), max_queue=0) as server:
        first = threading.Thread(target=lambda: request(server, "POST", "/v1/chat/completions", chat()))
        first.start()
        assert entered.wait(2)
        intervention = {"source_prompt":"source", "target_prompt":"target", "patch_layer":0}
        assert request(server, "POST", "/jspace/interventions", intervention)[0] == 503
        release.set()
        first.join(3)
        assert request(server, "POST", "/jspace/interventions", intervention)[0] == 200
