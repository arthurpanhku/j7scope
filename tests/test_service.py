"""HTTP-level reliability, privacy, and admission tests for the preview server."""
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from contextlib import closing, contextmanager

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


def test_event_surface_text_reconstructs_the_actual_chat_response():
    with running() as server:
        sub_id, events = server.bus.subscribe()
        try:
            code, _, raw = request(server, "POST", "/v1/chat/completions", chat(max_tokens=10))
            assert code == 200
            pieces = []
            while not events.empty():
                event = events.get_nowait()
                if event['type'] == 'token':
                    pieces.append(event['text'])
            expected = json.loads(raw)['choices'][0]['message']['content']
            assert ''.join(pieces) == expected
            assert ' ' in expected
        finally:
            server.bus.unsubscribe(sub_id)


def test_mood_embedding_neighbors_are_published_to_the_same_session():
    class Encoder:
        model_name = "local-test"

        def neighbors(self, text):
            assert text == "I feel hopeful."
            return [{"word": "hope", "cosine": 0.85}]

    with running(api_key=KEY, mood_embeddings=Encoder()) as server:
        session = json.loads(request(server, "POST", "/jspace/sessions", token=KEY)[2])
        sub, events = server.bus.subscribe(session=session["session_id"])
        try:
            data = {"source": "Codex", "text": "I feel hopeful.", "phase": "update"}
            assert request(server, "POST", "/mood/events", data, session["api_key"])[0] == 202
            event = events.get_nowait()
            assert event["phase"] == "update"
            assert event["embedding_model"] == "local-test"
            assert event["neighbors"] == [{"word": "hope", "cosine": 0.85}]
        finally:
            server.bus.unsubscribe(sub)


def test_tool_updates_require_a_session_bearer_and_validate_payload():
    sample = {"source": "External tool", "text": "开心", "phase": "update"}
    with running() as server:
        assert request(server, "POST", "/mood/events", sample)[0] == 403
    with running(api_key=KEY) as server:
        token = json.loads(request(server, "POST", "/jspace/sessions", token=KEY)[2])["api_key"]
        assert request(server, "POST", "/mood/events", sample)[0] == 401
        assert request(server, "POST", "/mood/events", sample, KEY)[0] == 403
        cookie = request(server, "POST", "/auth/login", token=token)[1]["Set-Cookie"].split(";", 1)[0]
        assert request(server, "POST", "/mood/events", sample, headers={"Cookie": cookie})[0] == 403
        for changes in ({"text": " "}, {"text": "开心" * 251}, {"text": "\ud800"},
                        {"source": "<b>tool</b>"}, {"phase": "unknown"}, {"prompt": "ignored"}):
            assert request(server, "POST", "/mood/events", sample | changes, token)[0] == 400
        assert request(server, "POST", "/mood/events", sample, token)[0] == 202


def test_stop_hooks_forward_only_reply_excerpt_to_the_authenticated_viewer(tmp_path):
    bridge = Path(__file__).resolve().parents[1] / "apps/serve/j7scope_serve/mood_bridge.py"
    with running(api_key=KEY, record_dir=tmp_path) as server:
        a, b = [json.loads(request(server, "POST", "/jspace/sessions", token=KEY)[2]) for _ in range(2)]
        cookie = request(server, "POST", "/auth/login", token=a["api_key"])[1]["Set-Cookie"].split(";", 1)[0]
        env = {**os.environ, "J7SCOPE_MOOD_SESSION_KEY": a["api_key"],
               "J7SCOPE_MOOD_PORT": str(server.server_port)}
        events = []
        with closing(http.client.HTTPConnection(*server.server_address, timeout=4)) as viewer:
            viewer.request("GET", "/jspace/stream", headers={"Cookie": cookie})
            response = viewer.getresponse()
            assert response.status == 200

            def read_event():
                line = response.readline()
                assert line.startswith(b"data: ")
                assert response.readline() == b"\n"
                return json.loads(line[6:])

            assert read_event()["tool_updates_enabled"] is True
            for mode in ("codex", "claude"):
                payload = {"hook_event_name": "Stop", "last_assistant_message": "a" * 550 + "开心",
                           "transcript_path": "/nonexistent/history", "input": "do not forward"}
                completed = subprocess.run([sys.executable, str(bridge), mode],
                                           input=json.dumps(payload), text=True,
                                           capture_output=True, env=env, timeout=4)
                assert (completed.returncode, completed.stdout.strip(), completed.stderr) == (0, "{}", "")
                events.append(read_event())
            response.close()
        assert [ev["source"] for ev in events] == ["Codex", "Claude Code"]
        for ev in events:
            assert ev["type"] == "mood_update" and ev["phase"] == "complete"
            assert ev["text"] == "a" * 498 + "开心"
            assert set(ev) == {"type", "id", "ts", "source", "text", "phase"}
        sid, other_events = server.bus.subscribe(session=b["session_id"])
        try:
            assert other_events.empty()
        finally:
            server.bus.unsubscribe(sid)
        assert not list(tmp_path.iterdir())


def test_stop_hook_is_nonblocking_when_input_or_local_service_is_unavailable():
    bridge = Path(__file__).resolve().parents[1] / "apps/serve/j7scope_serve/mood_bridge.py"
    for payload in ('{', '{"hook_event_name":"Stop","last_assistant_message":"开心"}'):
        completed = subprocess.run(
            [sys.executable, str(bridge), "codex"], input=payload, text=True,
            capture_output=True, timeout=3,
            env={**os.environ, "J7SCOPE_MOOD_PORT": "0", "J7SCOPE_MOOD_SESSION_KEY": "test-only"},
        )
        assert (completed.returncode, completed.stdout.strip(), completed.stderr) == (0, "{}", "")


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


def test_local_pairing_uses_single_use_ticket_and_scoped_cookie(tmp_path, monkeypatch):
    import secrets
    import shlex

    from j7scope_serve import mood_bridge, mood_integrate, mood_local

    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=0o700)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("J7SCOPE_MOOD_STATE_DIR", str(state_dir))
    monkeypatch.delenv("J7SCOPE_MOOD_SESSION_KEY", raising=False)
    monkeypatch.delenv("J7SCOPE_MOOD_PORT", raising=False)
    opened = []
    monkeypatch.setattr(mood_local.webbrowser, "open", lambda url: opened.append(url) or True)

    instance = secrets.token_urlsafe(32)
    with running(api_key=KEY, local_mood_id=instance) as server:
        key, principal = server.credentials.issue()

        def start(path):
            receiver = mood_local.Receiver(server.server_port, instance, key)
            path.write_text(json.dumps([receiver.port, instance, key]))
            path.chmod(0o600)
            return receiver, receiver.ticket()

        monkeypatch.setattr(mood_local, "_start", start)
        args = ["--project", str(project), "--agent", "codex", "--agent", "claude", "--apply", "--open"]
        assert mood_integrate.main(args) == 0
        monkeypatch.setattr(mood_local, "_start", lambda _: pytest.fail("receiver should be reused"))
        assert mood_integrate.main(args) == 0
        origin = f"http://127.0.0.1:{server.server_port}"
        prefix = origin + "/mood.html?mode=tools&view=compact&lang=en#pair="
        assert len(opened) == 2 and all(url.startswith(prefix) and key not in url for url in opened)
        body = {"ticket": opened[-1].removeprefix(prefix)}
        path = "/mood/pair/redeem"
        assert request(server, "POST", path, body, headers={"Origin": "https://example.com"})[0] == 403
        assert request(server, "POST", path, body, headers={"Origin": origin, "Host": "example.com"})[0] == 403
        status, headers, payload = request(server, "POST", path, body, headers={"Origin": origin})
        assert status == 200
        assert json.loads(payload)["session_id"] == principal.session
        assert "HttpOnly" in headers["Set-Cookie"] and "SameSite=Strict" in headers["Set-Cookie"]
        cookie = headers["Set-Cookie"].split(";", 1)[0]
        assert cookie.startswith(f"j7scope_mood_{server.server_port}=")
        assert request(server, "POST", path, body, headers={"Origin": origin})[0] == 401
        assert request(server, "POST", "/mood/pair", headers={"Cookie": cookie})[0] == 403
        assert request(server, "GET", "/jspace/session", headers={"Cookie": cookie})[0] == 200

        sub, events = server.bus.subscribe(session=principal.session)
        try:
            assert not mood_bridge.send_update(
                server.server_port, key, source="Codex", text="not sent", phase="complete",
                instance=secrets.token_urlsafe(32),
            )
            assert events.empty()
            for agent, expected in (("codex", "Codex"), ("claude", "Claude Code")):
                config = json.loads((project / mood_integrate.TARGETS[agent]).read_text())
                command = config["hooks"]["Stop"][0]["hooks"][0]["command"]
                assert key not in command
                payload = {"cwd": str(project), "hook_event_name": "Stop", "last_assistant_message": "happy"}
                subprocess.run(shlex.split(command), input=json.dumps(payload), text=True,
                               check=True, capture_output=True, timeout=4)
                event = events.get(timeout=2)
                assert (event["source"], event["text"], event["phase"]) == (expected, "happy", "complete")
        finally:
            server.bus.unsubscribe(sub)
