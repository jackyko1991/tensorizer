"""Completeness audit of the problem library: files, setup diagram, rendered output, translations, notebook.

python scripts/audit.py            # prints a table of problems with something missing; exit 1 if any
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import i18n  # noqa: E402
from i18n import hidden  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "frontend" / "public" / "generated"
CODE = ["meta.yaml", "main.py", "equation.py", "tensors.py", "solve.py", "plot.py"]


def audit() -> list[tuple[str, list[str]]]:
    rows = []
    for meta in sorted((ROOT / "problems").glob("*/*/meta.yaml")):
        if hidden(meta):
            continue
        d = meta.parent
        pid = f"{d.parent.name}/{d.name}"
        issues = [f"missing {f}" for f in CODE if not (d / f).is_file()]
        if not (d / "diagram.svg").is_file():
            issues.append("no setup diagram")
        out = GENERATED / d.parent.name / d.name
        if not any(p.suffix in {".gif", ".png"} and not p.name.startswith(".") for p in out.glob("*")):
            issues.append("no rendered animation")
        if not (ROOT / "notebooks" / d.parent.name / f"{d.name}.ipynb").is_file():
            issues.append("no notebook")
        if not issues or all(f.startswith("missing") is False for f in issues):
            en = i18n.english(d) if all((d / f).is_file() for f in CODE) else None
            if en:
                stale = [
                    lang
                    for lang in i18n.LANGS
                    if (tr := i18n.current(d, lang)) is None or tr.get("source_hash") != i18n.source_hash(en)
                ]
                if stale:
                    issues.append("translations missing/outdated: " + ",".join(stale))
        if issues:
            rows.append((pid, issues))
    return rows


if __name__ == "__main__":
    problems = [p for p in sorted((ROOT / "problems").glob("*/*/meta.yaml")) if not hidden(p)]
    rows = audit()
    print(f"{len(problems)} problems, {len(rows)} with something missing")
    for pid, issues in rows:
        print(f"  {pid:45s} {'; '.join(issues)}")
    sys.exit(1 if rows else 0)
