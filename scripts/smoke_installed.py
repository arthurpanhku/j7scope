"""Run with the installed package's Python, from outside the source checkout."""
import argparse
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-no-torch", action="store_true")
    args = parser.parse_args()
    if args.require_no_torch:
        assert importlib.util.find_spec("torch") is None, "base installation unexpectedly contains torch"
    with tempfile.TemporaryDirectory(prefix="j7scope-installed-") as work:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        cli = Path(sys.executable).parent / "j7scope"
        log = open(Path(work) / "service.log", "w+")
        process = subprocess.Popen([str(cli), "--backend", "mock", "--port", str(port),
            "--record", str(Path(work) / "traces"), "--token-delay", "0"],
            cwd=work, stdout=log, stderr=log)
        try:
            for _ in range(100):
                try:
                    urllib.request.urlopen(base + "/health", timeout=1).close()
                    break
                except (OSError, urllib.error.URLError):
                    if process.poll() is not None:
                        log.seek(0)
                        raise RuntimeError(log.read())
                    time.sleep(0.05)
            else:
                raise RuntimeError("installed service did not start")
            assert b"J7Scope" in urllib.request.urlopen(base + "/live.html").read()
            assert urllib.request.urlopen(base + "/assets/jspace.js").status == 200
            body = json.dumps({"messages": [{"role":"user","content":"hello"}], "max_tokens":4}).encode()
            response = urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions",
                body, {"Content-Type":"application/json"}))
            assert json.load(response)["choices"][0]["message"]["content"]
            index = json.load(urllib.request.urlopen(base + "/traces/index.json"))
            assert len(index["traces"]) == 1
            print(json.dumps({"passed":True,"cli":True,"bundled_viewer":True,"record_and_replay":True,
                              "torch_absent":importlib.util.find_spec("torch") is None}))
        finally:
            process.terminate()
            process.wait(timeout=5)
            log.close()


if __name__ == "__main__":
    main()
