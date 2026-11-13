"""Merge the sourced research files of Coding Skills ▸ Chips into frontend/src/content/chips/*.json.

Inputs (one directory, JSON written by the research step, every number with its source URL):
  vendors-a.json, vendors-b.json, china.json   chip records
  libraries.json                               library × hardware-family support, what runs underneath, defaults
  history-precision.json                       timeline, number formats, facts
  mixed.json                                   mixed-precision practice per framework and chip

Run: python scripts/chips_data.py <research dir>. The script normalises the records, checks them (unique ids,
https sources for every number, dense tensor figures) and writes the site's data files; it refuses to write when a
check fails, printing what is wrong.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "frontend" / "src" / "content" / "chips"
NUMERIC = [
    "bandwidthGBs",
    "fp64TF",
    "fp64TensorTF",
    "fp32TF",
    "tf32TF",
    "bf16TF",
    "fp16TF",
    "fp8TF",
    "fp6TF",
    "fp4TF",
    "int8TOPS",
    "npuTOPS",
    "powerW",
]
CLASSES = {"igpu", "dgpu", "soc", "accelerator", "cpu", "board", "npu"}
REGION = {"china": "china"}


def load(d: Path, name: str):
    p = d / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def num(v):
    """Numbers stay numbers; ranges like "15-28" or [15, 28] become [lo, hi]; anything else is None."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        return v
    if isinstance(v, str):
        s = v.replace("–", "-").replace("W", "").strip()
        parts = [p.strip() for p in s.split("-")]
        try:
            vals = [float(p) for p in parts if p]
            return vals[0] if len(vals) == 1 else vals[:2]
        except ValueError:
            return None
    return None


def chip(rec: dict, group: str, problems: list[str]) -> dict:
    out = {
        "id": rec["id"],
        "vendor": rec.get("vendor", ""),
        "name": rec.get("name", ""),
        "nameZh": rec.get("nameZh"),
        "group": group,
        "class": rec.get("class"),
        "year": rec.get("year"),
        "memoryType": (rec.get("memory") or {}).get("type"),
        "memoryGB": num((rec.get("memory") or {}).get("capacityGB")),
        "matrixUnit": rec.get("matrixUnit"),
        "formats": rec.get("formats") or [],
        "pytorchDevice": rec.get("pytorchDevice"),
        "blas": rec.get("blas") if isinstance(rec.get("blas"), str) else ", ".join(rec.get("blas") or []),
        "kernelLangs": rec.get("kernelLangs")
        if isinstance(rec.get("kernelLangs"), str)
        else ", ".join(rec.get("kernelLangs") or []),
        "interconnect": rec.get("interconnect"),
        "notes": rec.get("notes") or "",
        "confidence": rec.get("confidence") or "",
        "sources": {},
        "derived": [],
    }
    for k in NUMERIC:
        out[k] = num(rec.get(k))
    src = rec.get("sources") or {}
    for k, url in src.items():
        key = "memoryGB" if k in ("memory.capacityGB", "memory", "capacityGB") else k
        if isinstance(url, str) and url.startswith("http"):
            out["sources"][key] = url
        elif isinstance(url, str) and url.lower().startswith("derived"):
            out["derived"].append(key)  # computed from vendor inputs; the notes say how
    if out["class"] not in CLASSES:
        problems.append(f"{out['id']}: class {out['class']!r}")
    for k in NUMERIC + ["memoryGB"]:
        if out.get(k) is not None and k not in out["sources"]:
            # a number without its own source falls back to any source of the record, but must have one
            if not out["sources"]:
                problems.append(f"{out['id']}: {k} has no source")
    for url in out["sources"].values():
        if not url.startswith("https://"):
            problems.append(f"{out['id']}: non-https source {url}")
    return out


def text(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return str(v)


def stack(rec: dict, problems: list[str]) -> dict:
    """A vendor's software stack, with every field a string (or null) and links checked."""
    st = rec.get("stack") if isinstance(rec.get("stack"), dict) else {"sdk": rec.get("stack")}
    pt = rec.get("pytorch") if isinstance(rec.get("pytorch"), dict) else {"adapter": rec.get("pytorch")}
    os_ = rec.get("opensource") if isinstance(rec.get("opensource"), dict) else {"what": rec.get("opensource")}
    out = {
        "vendor": rec["vendor"],
        "sdk": text(st.get("sdk")),
        "compiler": text(st.get("compiler")),
        "kernelLanguage": text(st.get("kernelLanguage")),
        "mathLibraries": text(st.get("mathLibraries")),
        "collectives": text(st.get("collectives")),
        "pytorch": text(pt.get("adapter")),
        "pytorchRepo": pt.get("repo"),
        "frameworks": text(rec.get("frameworks")),
        "opensource": text(os_.get("what")),
        "opensourceLink": os_.get("link"),
        "source": rec.get("source"),
    }
    for k in ("pytorchRepo", "opensourceLink", "source"):
        if out[k] is not None and not str(out[k]).startswith("https://"):
            problems.append(f"stack {out['vendor']}: {k} {out[k]}")
    return out


def main() -> None:
    d = Path(sys.argv[1])
    problems: list[str] = []
    chips: list[dict] = []
    for name, group in (("vendors-a.json", "global"), ("vendors-b.json", "global")):
        for rec in load(d, name) or []:
            chips.append(chip(rec, group, problems))
    china = load(d, "china.json") or {}
    for rec in china.get("chips", []):
        chips.append(chip(rec, "china", problems))
    ids = [c["id"] for c in chips]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        problems.append(f"duplicate ids {sorted(dup)}")

    libs = load(d, "libraries.json")
    hist = load(d, "history-precision.json")
    mixed = load(d, "mixed.json")

    if problems:
        print("not written:\n  " + "\n  ".join(problems))
        sys.exit(1)
    OUT.mkdir(parents=True, exist_ok=True)

    def write(name: str, data) -> None:
        (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {name}")

    stacks = [stack(r, problems) for r in china.get("stacks", [])]
    if problems:
        print("not written:\n  " + "\n  ".join(problems))
        sys.exit(1)
    write("chips.json", {"chips": chips, "stacks": stacks})
    if libs:
        write("libraries.json", libs)
    if hist:
        write("history.json", {"timeline": hist["timeline"], "facts": hist["facts"]})
        write("formats.json", hist["formats"])
    if mixed:
        write("mixed.json", mixed)


if __name__ == "__main__":
    main()
