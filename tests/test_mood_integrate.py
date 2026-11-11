import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "serve"))

from j7scope_serve import mood_integrate as installer


def test_detect_preview_merge_and_remove_preserving_project_settings(tmp_path, monkeypatch):
    root = tmp_path / "project space"
    root.mkdir()
    (root / ".git").mkdir()
    (root / ".claude").mkdir()
    nested = root / "child"
    nested.mkdir()
    path = root / ".claude/settings.local.json"
    original = {"permissions": {"allow": ["Read"]},
                "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "existing-hook"}]}]}}
    path.write_text(json.dumps(original))
    path.chmod(0o640)
    monkeypatch.setattr(installer.shutil, "which", lambda name: "/test/bin/codex" if name == "codex" else None)
    args = ["--project", str(nested)]
    assert installer.discover(root) == ["codex", "claude"]
    assert installer.main(args) == 0
    assert not (root / ".codex").exists()
    assert path.read_text() == json.dumps(original)
    assert installer.main([*args, "--apply"]) == 0
    after = json.loads(path.read_text())
    assert after["permissions"] == original["permissions"]
    assert after["hooks"]["Stop"][0] == original["hooks"]["Stop"][0]
    command = shlex.split(after["hooks"]["Stop"][1]["hooks"][0]["command"])
    assert Path(command[1]).name == "mood_bridge.py" and command[2] == "claude"
    assert path.stat().st_mode & 0o777 == 0o640
    first = path.read_bytes()
    assert installer.main([*args, "--apply"]) == 0
    assert path.read_bytes() == first
    assert installer.main([*args, "--remove", "--apply"]) == 0
    assert json.loads(path.read_text()) == original
    assert json.loads((root / ".codex/hooks.json").read_text()) == {}


def test_opencode_can_install_and_remove_owned_plugin(tmp_path):
    args = ["--project", str(tmp_path), "--agent", "opencode", "--apply"]
    assert installer.main(args) == 0
    plugin = tmp_path / installer.TARGETS["opencode"]
    assert "J7ScopeMood" in plugin.read_text()
    assert installer.main([*args, "--remove"]) == 0
    assert not plugin.exists()
    plugin.write_text("// another plugin")
    assert installer.main(args) == 1
    assert installer.main([*args, "--remove"]) == 1
    assert plugin.read_text() == "// another plugin"


def test_invalid_json_and_symlink_do_not_change_other_configs(tmp_path):
    (tmp_path / ".claude").mkdir()
    target = tmp_path / ".claude/settings.local.json"
    target.write_text('{"hooks": {}, "hooks": {"Stop": []}}')
    args = ["--project", str(tmp_path), "--agent", "codex", "--agent", "claude", "--apply"]
    assert installer.main(args) == 1
    assert not (tmp_path / ".codex").exists()
    assert target.read_text().count('"hooks"') == 2
    target.unlink()
    other = tmp_path / "elsewhere.json"
    other.write_text("{}")
    target.symlink_to(other)
    assert installer.main(args) == 1
    assert other.read_text() == "{}"
    assert target.is_symlink()
