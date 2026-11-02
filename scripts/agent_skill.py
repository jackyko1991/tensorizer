"""The Tensorizer agent skill in every format, from one source: plugins/tensorizer/skills/tensorizer/SKILL.md.

    python scripts/agent_skill.py [--out frontend/public/generated]

writes <out>/agent/:
    SKILL.md               Claude Code skill (also installed by the plugin marketplace in .claude-plugin/)
    AGENTS.md              the same instructions for Codex, OpenCode and other agents that read AGENTS.md
    GEMINI.md              the same for Gemini CLI
    tensorizer-skill.zip   tensorizer/SKILL.md, to upload in Claude Desktop / claude.ai (Customize › Skills)

Called by build_assets.py, so the site always serves the current skill.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "plugins" / "tensorizer" / "skills" / "tensorizer" / "SKILL.md"
SITE = "https://jackyko1991.github.io/tensorizer/#/agents"


def body(skill: str) -> str:
    """SKILL.md without its YAML frontmatter."""
    if skill.startswith("---"):
        skill = skill.split("---", 2)[2]
    return skill.lstrip("\n")


def zip_bytes(skill: str) -> bytes:
    """A deterministic zip (fixed timestamps), so unchanged content gives identical bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        info = zipfile.ZipInfo("tensorizer/SKILL.md", date_time=(2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, skill)
    return buf.getvalue()


def files() -> dict[str, bytes]:
    skill = SOURCE.read_text(encoding="utf-8")
    note = f"<!-- Tensorizer agent instructions, generated from SKILL.md; see {SITE} -->\n\n"
    return {
        "SKILL.md": skill.encode(),
        "AGENTS.md": (note + body(skill)).encode(),
        "GEMINI.md": (note + body(skill)).encode(),
        "tensorizer-skill.zip": zip_bytes(skill),
    }


def write(out: Path) -> None:
    target = out / "agent"
    target.mkdir(parents=True, exist_ok=True)
    for name, data in files().items():
        path = target / name
        if not path.exists() or path.read_bytes() != data:
            path.write_bytes(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "frontend" / "public" / "generated"))
    args = parser.parse_args()
    write(Path(args.out).resolve())
    print(f"wrote {Path(args.out) / 'agent'}")
