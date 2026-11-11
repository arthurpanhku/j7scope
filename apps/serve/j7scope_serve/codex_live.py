"""Chat through Codex App Server and mirror the live reply to J7Scope."""

from __future__ import annotations

import argparse
from collections import deque
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .mood_bridge import send_update
from .mood_local import project_receiver


class AppServer:
    def __init__(self, process):
        self.process = process
        self.next_id = 0
        self.events = deque()

    def write(self, message):
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def read(self):
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("Codex App Server closed its stream")
            message = json.loads(line)
            if "method" in message and "id" in message:
                # This client is read-only and does not grant tool approvals.
                self.write({"id": message["id"], "error": {
                    "code": -32601, "message": "Client does not handle server requests",
                }})
                continue
            return message

    def request(self, method, params):
        self.next_id += 1
        self.write({"id": self.next_id, "method": method, "params": params})
        while True:
            message = self.read()
            if message.get("id") == self.next_id:
                if "error" in message:
                    raise RuntimeError(message["error"].get("message", "Codex request failed"))
                return message["result"]
            if "method" in message:
                self.events.append(message)

    def event(self):
        return self.events.popleft() if self.events else self.read()


def stream_reply(server, thread_id, turn_id, publish, display, clock=time.monotonic):
    """Keep only the latest assistant item's text and bound the mirror rate."""
    item_id = None
    text = ""
    last_update = None
    while True:
        event = server.event()
        method = event.get("method")
        params = event.get("params", {})
        if params.get("threadId") != thread_id:
            continue
        if method == "turn/completed" and params["turn"]["id"] == turn_id:
            status = params["turn"]["status"]
            if text.strip():
                publish(text, "complete" if status == "completed" else "interrupted")
            return status
        if params.get("turnId") != turn_id:
            continue
        if method == "item/agentMessage/delta":
            if item_id != params["itemId"]:
                if item_id is not None:
                    display("\n")
                item_id, text, last_update = params["itemId"], "", None
            delta = params["delta"]
            display(delta)
            text = (text + delta)[-500:]
            now = clock()
            if text.strip() and (last_update is None or now - last_update >= 0.25):
                publish(text, "update")
                last_update = now
        elif method == "item/completed" and params["item"]["type"] == "agentMessage":
            item = params["item"]
            if item["id"] != item_id:
                if item_id is not None:
                    display("\n")
                display(item["text"])
            item_id, text = item["id"], item["text"][-500:]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", help="send one prompt and exit")
    parser.add_argument("--model", help="override the Codex model for this session")
    parser.add_argument("--session-key-file", type=Path, help="read the J7Scope session key from this file")
    args = parser.parse_args(argv)
    try:
        key = os.environ.get("J7SCOPE_MOOD_SESSION_KEY", "")
        if args.session_key_file:
            key = args.session_key_file.read_text(encoding="utf-8").strip()
        port = int(os.environ.get("J7SCOPE_MOOD_PORT", "8799"))
        instance = None
        if not key and not args.session_key_file:
            receiver = project_receiver(os.getcwd())
            if receiver:
                key, port, instance = receiver.key, receiver.port, receiver.instance
        if not key or not 1 <= port <= 65535:
            raise ValueError("provide a J7Scope session key and a valid local port")
        failed = False

        def publish(text, phase):
            nonlocal failed
            try:
                accepted = send_update(port, key, source="Codex", text=text.strip(), phase=phase, instance=instance)
            except (OSError, ValueError, http.client.HTTPException):
                accepted = False
            if not accepted and not failed:
                print("\nDashboard update failed. Check the local service and session key.", file=sys.stderr)
            failed = not accepted

        with subprocess.Popen(["codex", "app-server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              text=True, encoding="utf-8") as process:
            try:
                server = AppServer(process)
                server.request("initialize", {"clientInfo": {
                    "name": "j7scope", "title": "J7Scope", "version": "0.2.0",
                }})
                server.write({"method": "initialized"})
                params = {"cwd": os.getcwd(), "approvalPolicy": "never", "sandbox": "read-only",
                          "ephemeral": True}
                if args.model:
                    params["model"] = args.model
                thread_id = server.request("thread/start", params)["thread"]["id"]
                print(f"Dashboard: http://127.0.0.1:{port}/mood.html?mode=tools&view=compact&lang=en")
                while True:
                    prompt = args.prompt if args.prompt is not None else input("\nYou: ")
                    if not prompt.strip():
                        return 0
                    turn = server.request("turn/start", {"threadId": thread_id, "input": [
                        {"type": "text", "text": prompt, "text_elements": []},
                    ]})["turn"]
                    print("Codex: ", end="", flush=True)
                    status = stream_reply(server, thread_id, turn["id"], publish,
                                          lambda text: print(text, end="", flush=True))
                    print()
                    if args.prompt is not None or status != "completed":
                        if status != "completed":
                            print(f"Codex turn ended with status {status}.", file=sys.stderr)
                        return int(status != "completed" or failed)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
    except (EOFError, KeyboardInterrupt):
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Codex live: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
