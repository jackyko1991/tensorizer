"""Paper links shown next to a problem's Wikipedia link (content/papers.json).

    python scripts/papers.py add <category/slug> <doi-or-arXiv-id> [...]   # verify online, then store
    python scripts/papers.py ref <category/slug> "<citation>" ["<title>"]   # a reference with no DOI/arXiv id
    python scripts/papers.py label <category/slug> <url> "<label>"           # fix a label (e.g. no author on record)
    python scripts/papers.py check                                         # re-verify every stored link

Every entry is looked up on Crossref (DOIs) or the arXiv API (arXiv ids) and its label (first author, year,
venue) and title are taken from that record, so a link can never point to a paper that doesn't exist.
A `ref` (e.g. a book or an old paper with no DOI) is shown as a plain citation without a link.
"""

from __future__ import annotations

import contextlib
import html
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "content" / "papers.json"
ARXIV = re.compile(r"^(arxiv:)?(\d{4}\.\d{4,5})(v\d+)?$", re.I)


def _get(url: str) -> str:
    # curl rather than urllib: it honours the system proxy/certificates on every platform we use.
    out = subprocess.run(
        ["curl", "-sfL", "-A", "tensorizer-papers/1.0 (https://github.com/jackyko1991/tensorizer)", url],
        capture_output=True,
        check=True,
    )
    return out.stdout.decode("utf-8")


def _clean(text: str) -> str:
    """Titles from Crossref may carry JATS/HTML markup (<i>…</i>) and entities."""
    return " ".join(html.unescape(re.sub(r"<[^>]+>", "", text)).split())


def _name(family: str) -> str:
    """'CLERK-MAXWELL' → 'Clerk-Maxwell'; mixed-case names are kept as they are."""
    return family.title() if family.isupper() else family


def lookup(ref: str) -> dict:
    """{label, title, url} for a DOI or arXiv id, from the publisher's metadata."""
    if m := ARXIV.match(ref.strip()):
        aid = m.group(2)
        xml = _get(f"https://export.arxiv.org/api/query?id_list={aid}")
        entry = xml.split("<entry>", 1)[1]
        title = _clean(re.search(r"<title>(.*?)</title>", entry, re.S).group(1))
        author = re.search(r"<name>(.*?)</name>", entry).group(1).split()[-1]
        year = re.search(r"<published>(\d{4})", entry).group(1)
        return {"label": f"{author} et al., {year} (arXiv)", "title": title, "url": f"https://arxiv.org/abs/{aid}"}
    doi = ref.strip().removeprefix("https://doi.org/")
    msg = json.loads(_get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"))["message"]
    authors = msg.get("author") or []
    first = _name(authors[0].get("family") or authors[0].get("name", "")) if authors else ""
    etal = " et al." if len(authors) > 1 else ""
    parts = (msg.get("published-print") or msg.get("published-online") or msg.get("issued"))["date-parts"][0]
    venue = (msg.get("short-container-title") or msg.get("container-title") or [""])[0]
    title = _clean((msg.get("title") or [""])[0])
    # No author on record (e.g. 17th–19th-century Royal Society papers): show the venue; `label` can fix it.
    label = ", ".join(x for x in (f"{first}{etal}" if first else "", str(parts[0]), venue) if x)
    return {"label": label, "title": title, "url": f"https://doi.org/{doi}"}


@contextlib.contextmanager
def _locked():
    """Several agents/processes may add papers at once: serialise read-modify-write with a lock file."""
    lock = PAPERS.with_suffix(".lock")
    for _ in range(600):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.time() - lock.stat().st_mtime > 120:  # a crashed writer left it behind
                lock.unlink(missing_ok=True)
            time.sleep(0.2)
    else:
        raise SystemExit(f"{lock} is held by another process")
    try:
        yield
    finally:
        os.close(fd)
        lock.unlink(missing_ok=True)


def _load() -> dict:
    return json.loads(PAPERS.read_text(encoding="utf-8")) if PAPERS.is_file() else {}


def _store(pid: str, found: list[dict]) -> None:
    data = _load()
    kept = [p for p in data.get(pid, []) if p["url"] not in {f["url"] for f in found}]
    data[pid] = kept + found
    _save(data)


def _save(data: dict) -> None:
    PAPERS.write_text(json.dumps(dict(sorted(data.items())), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    if len(sys.argv) >= 4 and sys.argv[1] == "add":
        pid, refs = sys.argv[2], sys.argv[3:]
        if not (ROOT / "problems" / pid / "meta.yaml").is_file():
            raise SystemExit(f"no problem {pid}")
        found = [lookup(r) for r in refs]  # network first, outside the lock
        with _locked():
            _store(pid, found)
        for p in found:
            print(f"{pid}: {p['label']} — {p['title']}  <{p['url']}>")
    elif len(sys.argv) in (4, 5) and sys.argv[1] == "ref":
        pid, label = sys.argv[2], sys.argv[3]
        title = sys.argv[4] if len(sys.argv) == 5 else ""
        if not (ROOT / "problems" / pid / "meta.yaml").is_file():
            raise SystemExit(f"no problem {pid}")
        with _locked():
            data = _load()
            entry = {"label": label, "title": title, "url": None}
            data[pid] = [p for p in data.get(pid, []) if p["label"] != label] + [entry]
            _save(data)
        print(f"{pid}: {label} (reference, no link)")
    elif len(sys.argv) == 5 and sys.argv[1] == "label":
        pid, url, label = sys.argv[2:]
        with _locked():
            data = _load()
            hits = [p for p in data.get(pid, []) if p["url"] == url]
            if not hits:
                raise SystemExit(f"{pid} has no paper {url}")
            hits[0]["label"] = label
            _save(data)
        print(f"{pid}: {url} → {label}")
    elif len(sys.argv) == 2 and sys.argv[1] == "check":
        for pid, papers in _load().items():
            for p in papers:
                if p["url"] is None:  # plain citation: nothing to look up
                    continue
                ref = p["url"].removeprefix("https://doi.org/").removeprefix("https://arxiv.org/abs/")
                assert lookup(ref)["title"] == _clean(p["title"]), f"{pid}: {p['url']} changed"
        print("all paper links verified")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
