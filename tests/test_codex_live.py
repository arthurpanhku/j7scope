from collections import deque
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "serve"))
from j7scope_serve.codex_live import stream_reply


def test_only_matching_assistant_text_is_streamed_and_finalized():
    def delta(text, *, thread="chat", turn="round", item="answer"):
        return {"method": "item/agentMessage/delta", "params": {
            "threadId": thread, "turnId": turn, "itemId": item, "delta": text,
        }}

    class Server:
        events = deque([
            delta("ignore", thread="another-chat"),
            delta("ignore", turn="old-round"),
            {"method": "item/reasoning/textDelta", "params": {
                "threadId": "chat", "turnId": "round", "delta": "private",
            }},
            delta("I am "),
            delta("happy"),
            delta(" and hopeful."),
            {"method": "item/completed", "params": {"threadId": "chat", "turnId": "round",
                 "item": {"type": "agentMessage", "id": "answer", "text": "I am happy and hopeful."}}},
            {"method": "turn/completed", "params": {"threadId": "chat",
                                                   "turn": {"id": "round", "status": "completed"}}},
        ])

        def event(self):
            return self.events.popleft()

    times = iter([1.0, 1.1, 1.3])
    pushes, output = [], []
    status = stream_reply(Server(), "chat", "round", lambda text, phase: pushes.append((text, phase)),
                          output.append, clock=lambda: next(times))
    assert status == "completed"
    assert "".join(output) == "I am happy and hopeful."
    assert pushes == [("I am ", "update"), ("I am happy and hopeful.", "update"),
                      ("I am happy and hopeful.", "complete")]
