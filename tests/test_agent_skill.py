"""The agent skill: valid frontmatter (what claude.ai accepts), the plugin marketplace, and the generated formats."""

import io
import json
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import agent_skill  # noqa: E402

UPLOAD_KEYS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}


def test_frontmatter_is_valid_for_claude_code_and_uploads():
    text = agent_skill.SOURCE.read_text(encoding="utf-8")
    front = yaml.safe_load(text.split("---", 2)[1])
    assert front["name"] == "tensorizer" and len(front["description"]) <= 1024
    assert set(front) <= UPLOAD_KEYS  # claude.ai / Desktop uploads reject other keys


def test_marketplace_points_at_the_plugin():
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
    entry = market["plugins"][0]
    plugin = ROOT / entry["source"] / ".claude-plugin" / "plugin.json"
    assert json.loads(plugin.read_text(encoding="utf-8"))["name"] == entry["name"] == "tensorizer"
    assert (ROOT / entry["source"] / "skills" / "tensorizer" / "SKILL.md").samefile(agent_skill.SOURCE)


def test_generated_formats():
    files = agent_skill.files()
    assert not files["AGENTS.md"].decode().lstrip("<!-").startswith("---")  # no frontmatter for AGENTS.md readers
    assert b"# The Tensorizer method" in files["AGENTS.md"] and files["AGENTS.md"] == files["GEMINI.md"]
    names = zipfile.ZipFile(io.BytesIO(files["tensorizer-skill.zip"])).namelist()
    assert names == ["tensorizer/SKILL.md"]  # one skill folder per zip
    assert agent_skill.files() == files  # deterministic bytes
