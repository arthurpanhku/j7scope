"""Forward a short assistant reply excerpt to the local J7Scope session."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys

MAX_INPUT = 2_000_000
HOOK_SOURCES = {"codex": "Codex", "claude": "Claude Code"}


def message_from_input(mode: str, raw: bytes) -> str:
    if len(raw) > MAX_INPUT:
        return ""
    if mode == "text":
        message = raw.decode("utf-8")
    else:
        event = json.loads(raw)
        if not isinstance(event, dict) or event.get("hook_event_name") != "Stop":
            return ""
        message = event.get("last_assistant_message")
    return message.strip()[-500:].strip() if isinstance(message, str) else ""


def send_update(port: int, key: str, *, source: str, text: str, phase: str) -> bool:
    # A literal loopback address and HTTPConnection avoid proxies and redirects.
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
    try:
        connection.request("POST", "/mood/events",
                           json.dumps({"source": source, "text": text, "phase": phase}),
                           {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        with connection.getresponse() as response:
            return response.status == 202
    finally:
        connection.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=(*HOOK_SOURCES, "text"))
    parser.add_argument("--source", default="External tool", help="tool name for text mode")
    parser.add_argument("--phase", choices=("update", "complete"), default="complete")
    args = parser.parse_args(argv)
    is_hook = args.mode in HOOK_SOURCES
    accepted = False
    try:
        text = message_from_input(args.mode, sys.stdin.buffer.read(MAX_INPUT + 1))
        key = os.environ.get("J7SCOPE_MOOD_SESSION_KEY", "")
        port = int(os.environ.get("J7SCOPE_MOOD_PORT", "8799"))
        if key and text and 1 <= port <= 65535:
            accepted = send_update(port, key, source=HOOK_SOURCES.get(args.mode, args.source),
                                   text=text, phase="complete" if is_hook else args.phase)
    except (ValueError, UnicodeError, RecursionError, OSError, http.client.HTTPException):
        pass
    if is_hook:
        # Codex Stop expects JSON. An unavailable viewer must not block either tool.
        print("{}")
        return 0
    if not accepted:
        print("Mood update not accepted. Check the local service, session key and input.", file=sys.stderr)
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
