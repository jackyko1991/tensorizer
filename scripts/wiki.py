"""Wikipedia links shown under each problem's sketch (content/wiki.json).

    python scripts/wiki.py add <category/slug> "<English Wikipedia article title>"

The title is resolved on en.wikipedia (following redirects), and the article's Wikidata sitelinks give its title in
every supported language, so each link points to a real article in the reader's language when one exists.
"""

from __future__ import annotations

import json
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from papers import _get, _locked  # noqa: E402  (same curl helper and write lock)

ROOT = Path(__file__).resolve().parents[1]
WIKI = ROOT / "content" / "wiki.json"
LANGS = ["en", "zh", "fr", "de", "es", "ja", "ru", "ko", "hi"]


def lookup(title: str) -> dict:
    q = json.loads(
        _get(
            "https://en.wikipedia.org/w/api.php?action=query&format=json&redirects=1&prop=pageprops&titles="
            + urllib.parse.quote(title)
        )
    )["query"]
    page = next(iter(q["pages"].values()))
    if "missing" in page or "pageprops" not in page or "disambiguation" in page["pageprops"]:
        raise SystemExit(f"no usable English Wikipedia article {title!r}")
    qid = page["pageprops"]["wikibase_item"]
    links = json.loads(
        _get(f"https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&props=sitelinks&ids={qid}")
    )["entities"][qid]["sitelinks"]
    titles = {lang: links[f"{lang}wiki"]["title"] for lang in LANGS if f"{lang}wiki" in links}
    return {"titles": titles, "wikidata": qid}


def main() -> None:
    if len(sys.argv) != 4 or sys.argv[1] != "add":
        raise SystemExit(__doc__)
    pid, title = sys.argv[2], sys.argv[3]
    if not (ROOT / "problems" / pid / "meta.yaml").is_file():
        raise SystemExit(f"no problem {pid}")
    entry = lookup(title)
    with _locked():
        data = json.loads(WIKI.read_text(encoding="utf-8")) if WIKI.is_file() else {}
        data[pid] = entry
        WIKI.write_text(json.dumps(dict(sorted(data.items())), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{pid}: {entry['titles'].get('en')} ({entry['wikidata']}, {len(entry['titles'])} languages)")


if __name__ == "__main__":
    main()
