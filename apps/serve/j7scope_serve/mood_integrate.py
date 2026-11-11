"""Discover and configure project-local J7Scope mood integrations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import shutil
import sys
import tempfile

TARGETS = {
    "codex": Path(".codex/hooks.json"),
    "claude": Path(".claude/settings.local.json"),
    "opencode": Path(".opencode/plugins/j7scope-mood.js"),
}
MARKERS = {"codex": (".codex",), "claude": (".claude", "CLAUDE.md"),
           "opencode": (".opencode", "opencode.json", "opencode.jsonc")}


def discover(root: Path) -> list[str]:
    return [name for name in TARGETS if shutil.which(name) or any((root / p).exists() for p in MARKERS[name])]


def project_root(path: Path) -> Path:
    root = path.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("project must be a directory")
    return next((p for p in (root, *root.parents) if (p / ".git").exists()), root)


def read_target(root: Path, relative: Path) -> tuple[Path, bytes | None]:
    path = root
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            raise ValueError(f"refusing to modify symlink: {path}")
    if path.exists() and not path.is_file():
        raise ValueError(f"expected a file: {path}")
    return path, path.read_bytes() if path.is_file() else None


def hook_entry(agent: str) -> dict:
    bridge = Path(__file__).resolve().with_name("mood_bridge.py")
    # Codex and Claude Code execute command hooks through a shell.
    command = shlex.join([sys.executable, str(bridge), agent])
    return {"hooks": [{"type": "command", "command": command, "timeout": 3}]}


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def is_bridge(command, agent: str) -> bool:
    if not isinstance(command, str):
        return False
    try:
        parts = shlex.split(command)
    except ValueError:
        return False
    if len(parts) != 3 or parts[2] != agent or not Path(parts[1]).is_absolute():
        return False
    return Path(parts[1]).resolve() == Path(__file__).resolve().with_name("mood_bridge.py")


def plan(root: Path, agent: str, remove: bool) -> tuple[Path, bytes | None, bytes | None]:
    path, before = read_target(root, TARGETS[agent])
    if agent == "opencode":
        plugin = Path(__file__).with_name("mood_opencode.mjs").read_bytes()
        if before is not None and before != plugin:
            raise ValueError(f"existing plugin differs; leaving it unchanged: {path}")
        return path, before, None if remove else plugin
    if remove and before is None:
        return path, before, before
    try:
        config = json.loads(before, object_pairs_hook=unique_keys, parse_constant=invalid_constant) if before is not None else {}
    except (ValueError, RecursionError):
        raise ValueError(f"invalid JSON; leaving it unchanged: {path}") from None
    if not isinstance(config, dict) or not isinstance(config.get("hooks", {}), dict):
        raise ValueError(f"unsupported hooks structure: {path}")
    hooks = config.get("hooks", {})
    stop = hooks.get("Stop", [])
    if not isinstance(stop, list):
        raise ValueError(f"hooks.Stop must be a list: {path}")
    entry = hook_entry(agent)
    # Match the command as well, to avoid duplicates if the user adjusted timeout.
    if remove:
        if entry not in stop:
            return path, before, before
        stop = [group for group in stop if group != entry]
        if stop:
            hooks["Stop"] = stop
        else:
            hooks.pop("Stop")
        if not hooks:
            config.pop("hooks")
    else:
        if any(is_bridge(handler.get("command"), agent) for group in stop if isinstance(group, dict)
               for handler in (group.get("hooks") if isinstance(group.get("hooks"), list) else [])
               if isinstance(handler, dict)):
            return path, before, before
        hooks["Stop"] = [*stop, entry]
        config["hooks"] = hooks
    after = (json.dumps(config, ensure_ascii=False, indent=2) + "\n").encode()
    return path, before, after


def apply_change(path: Path, before: bytes | None, after: bytes | None) -> None:
    if before == after:
        return
    if path.is_symlink() or (path.read_bytes() if path.is_file() else None) != before:
        raise ValueError(f"file changed during setup: {path}")
    if after is None:
        path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".j7scope-", delete=False) as stream:
        temp = Path(stream.name)
        try:
            stream.write(after)
            stream.close()
            if before is not None:
                temp.chmod(path.stat().st_mode & 0o777)
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd(), help="project folder (defaults to current checkout)")
    parser.add_argument("--agent", choices=TARGETS, action="append", help="override detection; repeat for multiple tools")
    parser.add_argument("--apply", action="store_true", help="write changes; without this flag, only show the plan")
    parser.add_argument("--remove", action="store_true", help="remove unchanged J7Scope entries")
    args = parser.parse_args(argv)
    if sys.platform == "win32":
        parser.error("automatic command configuration currently supports macOS and Linux")
    try:
        root = project_root(args.project)
        agents = list(dict.fromkeys(args.agent or discover(root)))
        if not agents:
            print("No supported agent found. Use --agent codex, claude, or opencode to select one.")
            return 0
        changes = [(name, *plan(root, name, args.remove)) for name in agents]
        for agent, path, before, after in changes:
            action = "unchanged" if before == after else "remove" if after is None else "create" if before is None else "update"
            print(f"{agent}: {action} {path}")
            if args.apply:
                apply_change(path, before, after)
        if not args.apply:
            print("Preview only. Add --apply to change project files.")
        elif not args.remove:
            print("Restart the agents with J7SCOPE_MOOD_SESSION_KEY set. Review Codex hooks with /hooks.")
        return 0
    except (OSError, ValueError) as exc:
        print(f"Integration setup: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
