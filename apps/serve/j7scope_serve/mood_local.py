"""Manage the local dashboard receiver and discover its credentials from hooks."""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import hashlib
import http.client
import json
import os
from pathlib import Path
import secrets
import stat
import subprocess
import sys
import tempfile
import time
import webbrowser


@dataclass(frozen=True)
class Receiver:
    port: int
    instance: str
    key: str

    def ticket(self):
        with closing(http.client.HTTPConnection("127.0.0.1", self.port, timeout=1)) as connection:
            if not matches_receiver(connection, self.instance):
                raise ConnectionError("local receiver was replaced")
            connection.request("POST", "/mood/pair", headers={"Authorization": f"Bearer {self.key}"})
            with connection.getresponse() as response:
                if response.status == 401:
                    raise ConnectionError("local receiver session expired")
                if response.status != 201:
                    raise ValueError("local receiver could not pair the dashboard")
                return json.loads(response.read())["ticket"]


def matches_receiver(connection, instance: str) -> bool:
    connection.request("GET", "/health")
    with connection.getresponse() as response:
        if response.status != 200 or response.will_close:
            return False
        try:
            health = json.loads(response.read(1024))
        except ValueError:
            return False
        return health == {"ok": True, "service": "j7scope", "local_mood_id": instance}


def state_path(project: Path) -> Path:
    home = Path(os.environ.get("J7SCOPE_MOOD_STATE_DIR") or Path.home() / ".j7scope-mood")
    fingerprint = hashlib.sha256(os.fsencode(project)).hexdigest()
    return home / f"{fingerprint}.json"


def read_receiver(path: Path) -> Receiver | None:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd) as stream:
            info = os.fstat(stream.fileno())
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
                return None
            if not stat.S_ISREG(info.st_mode):
                return None
            port, instance, key = json.load(stream)
        if type(port) is int and 1 <= port <= 65535 and all(
            isinstance(value, str) and len(value) == 43 and value.isascii() for value in (instance, key)
        ):
            return Receiver(port, instance, key)
    except (OSError, ValueError, TypeError):
        pass
    return None


def project_receiver(cwd: str) -> Receiver | None:
    path = Path(cwd)
    if os.name != "posix" or not path.is_absolute():
        return None
    try:
        path = path.resolve()
    except RuntimeError:
        return None
    for directory in (path, *path.parents):
        receiver = read_receiver(state_path(directory))
        if receiver or (directory / ".git").exists():
            return receiver
    return None


def _start(path: Path) -> tuple[Receiver, str]:
    # Invoke the installed package from its own directory, including in source checkouts.
    process = subprocess.Popen(
        [sys.executable, "-m", "j7scope_serve.mood_local", str(path)],
        cwd=Path(__file__).resolve().parent.parent,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + 8
    try:
        while time.monotonic() < deadline and process.poll() is None:
            receiver = read_receiver(path)
            if receiver:
                try:
                    return receiver, receiver.ticket()
                except (OSError, ValueError, http.client.HTTPException) as exc:
                    raise ValueError("local mood receiver started but could not pair") from exc
            time.sleep(0.05)
        raise ValueError("local mood receiver did not start")
    except BaseException:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=3)
        raise


def open_dashboard(project: Path) -> bool:
    import fcntl

    path = state_path(project)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.parent.stat()
    if path.parent.is_symlink() or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise ValueError(f"mood state directory must be private: {path.parent}")
    lock = os.open(path.with_suffix(".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(lock, "r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        receiver = read_receiver(path)
        ticket = None
        if receiver:
            try:
                ticket = receiver.ticket()
            except (OSError, http.client.HTTPException):
                pass
        if not ticket:
            path.unlink(missing_ok=True)
            receiver, ticket = _start(path)
    url = f"http://127.0.0.1:{receiver.port}/mood.html?mode=tools&view=compact&lang=en#pair={ticket}"
    return webbrowser.open(url)


def _serve(path: Path) -> None:
    from threading import Timer

    from .app import SidecarServer
    from .backends import MockBackend

    instance = secrets.token_urlsafe(32)
    operator_key = secrets.token_urlsafe(32)
    with SidecarServer(("127.0.0.1", 0), MockBackend(), api_key=operator_key,
                       local_mood_id=instance, session_ttl=7 * 86400) as server:
        key, _ = server.credentials.issue()
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                json.dump([server.server_port, instance, key], stream)
                stream.close()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        # The sender credential lives for the lifetime of this receiver.
        expiry = Timer(server.credentials.ttl, server.shutdown)
        expiry.daemon = True
        expiry.start()
        try:
            server.serve_forever()
        finally:
            expiry.cancel()


if __name__ == "__main__":
    _serve(Path(sys.argv[1]))
