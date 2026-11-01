"""Problem-content translations: export what needs translating, import translations, show coverage.

    python scripts/i18n.py status
    python scripts/i18n.py export --lang ja --out ja.json     # missing or outdated items, English text
    python scripts/i18n.py import --lang ja ja.json           # validate and write problems/*/*/i18n/ja.yaml

Each translation file stores `source_hash`, a hash of the English fields it was made from, so editing the
English marks it outdated. Import refuses a translation whose inline math ($...$ spans) differs from the
English, or whose meanings/labels don't line up with the problem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ROOT / "problems"
LANGS = ["zh-Hant", "zh-Hans", "fr", "de", "es", "ja", "ru", "ko"]
FILES = ["meta.yaml", "main.py", "equation.py", "tensors.py", "solve.py", "plot.py"]
MATH = re.compile(r"(?<!\\)\$(.+?)(?<!\\)\$", re.S)


def problems() -> list[Path]:
    return sorted(p.parent for p in PROBLEMS.glob("*/*/meta.yaml") if all((p.parent / f).is_file() for f in FILES))


def english(problem: Path) -> dict:
    meta = yaml.safe_load((problem / "meta.yaml").read_text(encoding="utf-8"))
    return {
        "title": meta["title"],
        "summary": " ".join(str(meta.get("summary", "")).split()),
        "notes": " ".join(str(meta.get("notes", "")).split()),
        "meanings": [row.get("meaning", "") for row in meta.get("translation", [])],
        "labels": {k: v.get("label", k) for k, v in meta.get("params", {}).items()},
    }


def source_hash(fields: dict) -> str:
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def current(problem: Path, lang: str) -> dict | None:
    f = problem / "i18n" / f"{lang}.yaml"
    return yaml.safe_load(f.read_text(encoding="utf-8")) if f.is_file() else None


def pid(problem: Path) -> str:
    return f"{problem.parent.name}/{problem.name}"


def todo(lang: str) -> dict[str, dict]:
    out = {}
    for p in problems():
        en = english(p)
        tr = current(p, lang)
        if not tr or tr.get("source_hash") != source_hash(en):
            out[pid(p)] = en
    return out


def check(en: dict, tr: dict) -> list[str]:
    errors = []
    for key in ("title", "summary", "notes"):
        if en[key] and not str(tr.get(key, "")).strip():
            errors.append(f"{key} missing")
        elif sorted(MATH.findall(en[key])) != sorted(MATH.findall(str(tr.get(key, "")))):
            errors.append(f"{key}: inline math differs from English")
    if len(tr.get("meanings", [])) != len(en["meanings"]):
        errors.append(f"meanings: expected {len(en['meanings'])}, got {len(tr.get('meanings', []))}")
    else:
        for i, (a, b) in enumerate(zip(en["meanings"], tr["meanings"], strict=True)):
            if sorted(MATH.findall(a)) != sorted(MATH.findall(b)):
                errors.append(f"meanings[{i}]: inline math differs from English")
    if set(tr.get("labels", {})) != set(en["labels"]):
        errors.append("labels: keys differ from the parameters")
    else:
        for k, v in tr["labels"].items():
            if sorted(MATH.findall(en["labels"][k])) != sorted(MATH.findall(v)):
                errors.append(f"labels.{k}: inline math differs from English")
    return errors


def cmd_status(_args) -> None:
    ps = problems()
    print(f"{len(ps)} problems")
    for lang in LANGS:
        missing = todo(lang)
        print(f"  {lang:8s} {len(ps) - len(missing):3d} up to date, {len(missing):3d} missing/outdated")


def cmd_export(args) -> None:
    items = todo(args.lang)
    Path(args.out).write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"exported {len(items)} item(s) for {args.lang} -> {args.out}")


def cmd_import(args) -> None:
    data = json.loads(Path(args.file).read_text(encoding="utf-8"))
    by_id = {pid(p): p for p in problems()}
    written, failed = 0, 0
    for problem_id, tr in data.items():
        if problem_id not in by_id:
            print(f"skip {problem_id}: unknown problem")
            failed += 1
            continue
        problem = by_id[problem_id]
        en = english(problem)
        errors = check(en, tr)
        if errors:
            print(f"FAIL {problem_id}: " + "; ".join(errors))
            failed += 1
            continue
        out = {
            "source_hash": source_hash(en),
            **{k: tr[k] for k in ("title", "summary", "notes", "meanings", "labels")},
        }
        (problem / "i18n").mkdir(exist_ok=True)
        text = yaml.safe_dump(out, allow_unicode=True, sort_keys=False, width=1000)
        (problem / "i18n" / f"{args.lang}.yaml").write_text(text, encoding="utf-8")
        written += 1
    print(f"{args.lang}: wrote {written}, failed {failed}")
    if failed:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    e = sub.add_parser("export")
    e.add_argument("--lang", required=True, choices=LANGS)
    e.add_argument("--out", required=True)
    e.set_defaults(fn=cmd_export)
    i = sub.add_parser("import")
    i.add_argument("--lang", required=True, choices=LANGS)
    i.add_argument("file")
    i.set_defaults(fn=cmd_import)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
