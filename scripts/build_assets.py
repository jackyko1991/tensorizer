"""Render every problem with its default parameters and write the frontend manifest.

    python scripts/build_assets.py [--out frontend/public/generated] [--only mechanics/projectile]

Output layout:
    <out>/manifest.json                    categories, problem metadata and the source of the 5 code files
    <out>/<category>/<slug>/<name>.gif     prebuilt result (or .png for static problems)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ROOT / "problems"
CODE_FILES = ["main.py", "equation.py", "tensors.py", "solve.py", "plot.py"]
LEVELS = ["beginner", "intermediate", "advanced"]
CATEGORIES = {
    "mechanics": "Mechanics",
    "optics": "Optics",
    "waves": "Waves",
    "thermodynamics": "Thermodynamics",
    "em": "Electromagnetism",
    "fluids": "Fluids & Continuum",
    "quantum": "Quantum",
    "astro": "Astrophysics & Relativity",
    "nuclear": "Nuclear & Particle",
}


def complete(d: Path) -> bool:
    """Skip problems that are still being written (missing files or unreadable meta.yaml)."""
    try:
        yaml.safe_load((d / "meta.yaml").read_text(encoding="utf-8"))["level"]
    except Exception:
        return False
    return all((d / f).is_file() for f in CODE_FILES)


def discover() -> list[Path]:
    dirs = [p.parent for p in PROBLEMS.glob("*/*/meta.yaml")]
    for d in dirs:
        if not complete(d):
            print(f"skipping incomplete problem {d.parent.name}/{d.name}")
    dirs = [d for d in dirs if complete(d)]
    order = list(CATEGORIES)

    def key(d: Path):
        meta = yaml.safe_load((d / "meta.yaml").read_text(encoding="utf-8"))
        cat = d.parent.name
        level = LEVELS.index(meta["level"]) if meta.get("level") in LEVELS else len(LEVELS)
        return (order.index(cat) if cat in order else len(order), cat, level, meta.get("order", 99))

    return sorted(dirs, key=key)


def stale(problem: Path, target: Path) -> bool:
    """True when there is no output yet or any problem/_common source is newer than it."""
    outputs = [p for p in target.glob("*") if p.suffix in {".gif", ".png"} and not p.name.startswith(".")]
    if not outputs:
        return True
    sources = [*problem.glob("*.py"), problem / "meta.yaml", *(PROBLEMS / "_common").glob("*.py")]
    return max(f.stat().st_mtime for f in sources) > min(o.stat().st_mtime for o in outputs)


def load_translations(problem: Path) -> dict:
    """problems/<cat>/<slug>/i18n/<lang>.yaml → {lang: {title, summary, notes, meanings, labels}}."""
    out = {}
    for f in sorted((problem / "i18n").glob("*.yaml")):
        try:
            out[f.stem] = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            print(f"warning: skipping {f}: {e}")
    return out


def copy_diagram(problem: Path, target: Path) -> str | None:
    """Copy the problem's hand-drawn diagram.svg next to its output; return its manifest path."""
    src, dst = problem / "diagram.svg", target / "diagram.svg"
    if not src.is_file():
        return None
    if not dst.is_file() or dst.read_bytes() != src.read_bytes():
        try:
            shutil.copyfile(src, dst)
        except PermissionError:  # Windows: the old copy is being served right now; keep it this round
            pass
    return f"{problem.parent.name}/{problem.name}/diagram.svg?v={int(src.stat().st_mtime)}"


def render(problem: Path, out_dir: Path) -> tuple[Path, str]:
    module = f"problems.{problem.parent.name}.{problem.name}.main"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [sys.executable, "-m", module, "--out", str(out_dir)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{module} failed:\n{proc.stderr}")
    lines = proc.stdout.strip().splitlines()
    output = Path(json.loads(lines[-1])["output"])
    return output, "\n".join(lines[:-1])


def timed_render(problem: Path, out_dir: Path) -> tuple[Path, str, float]:
    start = time.perf_counter()
    output, console = render(problem, out_dir)
    return output, console, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "frontend" / "public" / "generated"))
    parser.add_argument("--only", help="category/slug to rebuild (manifest still lists every problem)")
    parser.add_argument("--force", action="store_true", help="re-render even if the output is up to date")
    parser.add_argument("--manifest-only", action="store_true", help="refresh manifest.json without rendering")
    parser.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="parallel renders")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    dirs = discover()
    # Share the cores between parallel renders instead of every torch process grabbing all of them.
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 2) // args.jobs)))

    # Render what is stale in parallel: each render is its own Python process.
    todo = []
    for d in dirs:
        target = out / d.parent.name / d.name
        target.mkdir(parents=True, exist_ok=True)
        if args.only in (None, f"{d.parent.name}/{d.name}") and (args.force or args.only or stale(d, target)):
            todo.append((d, target))
    if todo and not args.manifest_only:
        print(f"rendering {len(todo)} problem(s) with {args.jobs} job(s)")
        with ThreadPoolExecutor(args.jobs) as pool:
            futures = {pool.submit(timed_render, d, target): (d, target) for d, target in todo}
            for future in as_completed(futures):
                d, target = futures[future]
                output, console, seconds = future.result()
                (target / "console.txt").write_text(console, encoding="utf-8")
                print(f"{d.parent.name}/{d.name}: {output.name} in {seconds:.1f}s", flush=True)

    problems = []
    for d in dirs:
        cat, slug = d.parent.name, d.name
        meta = yaml.safe_load((d / "meta.yaml").read_text(encoding="utf-8"))
        target = out / cat / slug

        existing = sorted(p for p in target.iterdir() if p.suffix in {".gif", ".png"} and not p.name.startswith("."))
        console_file = target / "console.txt"
        problems.append(
            {
                "id": f"{cat}/{slug}",
                "category": cat,
                "slug": slug,
                **meta,
                "files": [{"name": f, "code": (d / f).read_text(encoding="utf-8")} for f in CODE_FILES],
                # ?v=mtime busts the browser cache when a GIF is re-rendered under the same name
                "output": f"{cat}/{slug}/{existing[0].name}?v={int(existing[0].stat().st_mtime)}" if existing else None,
                "console": console_file.read_text(encoding="utf-8") if console_file.exists() else "",
                "diagram": copy_diagram(d, target),
                "i18n": load_translations(d),
            }
        )

    categories = [{"id": c, "title": t} for c, t in CATEGORIES.items() if any(p["category"] == c for p in problems)]
    manifest = {"categories": categories, "levels": LEVELS, "problems": problems}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out / 'manifest.json'} ({len(problems)} problems)")


if __name__ == "__main__":
    main()
