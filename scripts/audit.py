"""Completeness audit of the problem library: files, setup diagram, rendered output, translations, notebook, and
the source and licence of every bundled data file.

python scripts/audit.py            # prints a table of problems with something missing; exit 1 if any
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import i18n  # noqa: E402
from i18n import hidden  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "frontend" / "public" / "generated"
CODE = ["meta.yaml", "test.py", "main.py", "equation.py", "tensors.py", "solve.py", "plot.py"]

TEXT_DATA = {".csv", ".tsv", ".txt", ".dat", ".json", ".md"}
SOURCE = re.compile(r"source|https?://|doi[:.]|arxiv|generated|synthetic|computed|derived from", re.I)
LICENCE = re.compile(
    r"licen[cs]e|public domain|\bpd\b|cc0|cc[ -]by|odbl|\bmit\b|\bbsd\b|apache|generated|synthetic|computed"
    r"|free use|terms of use|conditions of use|cited|facts",
    re.I,
)


def data_source_issues(d: Path) -> list[str]:
    """Bundled data files that don't say where they come from and under what licence.

    A text file says so in its leading # lines. A binary file (STL, image, array) is covered by a sidecar
    <name>.source(.txt) or by a data/SOURCES.md (or README.md) that names it."""
    root = d / "data"
    if not root.is_dir():
        return []
    sidecar = next((root / n for n in ("SOURCES.md", "README.md") if (root / n).is_file()), None)
    listing = sidecar.read_text(encoding="utf-8", errors="replace") if sidecar else ""
    bad = []
    for f in sorted(root.rglob("*")):
        rel = f.relative_to(root)
        if not f.is_file() or any(part.startswith((".", "__")) for part in rel.parts) or f == sidecar:
            continue
        if f.suffix in {".source", ".txt"} and f.stem.endswith(tuple(TEXT_DATA | {".stl", ".png", ".npy"})):
            continue  # a sidecar itself
        if f.suffix.lower() in TEXT_DATA:
            head = "\n".join(
                line
                for line in f.read_text(encoding="utf-8", errors="replace").splitlines()[:40]
                if line.startswith("#")
            )
            head += "\n" + (listing if f.name in listing else "")
        else:
            side = next(
                (
                    f.with_name(f.name + ext)
                    for ext in (".source", ".source.txt")
                    if f.with_name(f.name + ext).is_file()
                ),
                None,
            )
            head = (
                side.read_text(encoding="utf-8", errors="replace") if side else (listing if f.name in listing else "")
            )
        if not (SOURCE.search(head) and LICENCE.search(head)):
            bad.append(rel.as_posix())
    return [f"data without source/licence: {', '.join(bad)}"] if bad else []


def audit() -> list[tuple[str, list[str]]]:
    rows = []
    content = ROOT / "content"
    wiki, papers = (
        {} if not f.is_file() else json.loads(f.read_text(encoding="utf-8"))
        for f in (
            content / "wiki.json",
            content / "papers.json",
        )
    )
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
        if pid not in wiki:
            issues.append("no Wikipedia link (content/wiki.json)")
        if not papers.get(pid):
            issues.append("no paper or reference (scripts/papers.py)")
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
        issues += data_source_issues(d)
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
