"""Make the bundled subsets from the downloaded originals (run once, offline; read with numpy, PIL and csv only):

    python -I data/make_subset.py nih <Data_Entry_2017_v2020.csv> <folder of NIH PNGs> data/
    python -I data/make_subset.py pmc <records.jsonl> <folder of PMC figure files> data/ <sentences.jsonl>

nih: every PNG fetched (one frontal PA image per patient, findings among the six kept, see SOURCES.md) with at most
two of the six findings → 64×64 grayscale uint8 (box filter); split stratified by the first finding, every 4th of a
shuffled class (numpy default_rng(0)) to the test set. Writes nih_cxr_subset.npz with images (N,64,64) uint8,
labels (N,) int8 (0 = no finding, 1 + index of the first finding), findings (N,6) uint8 multi-hot, split (N,) uint8
(0 train, 1 test), image_index (N,) the NIH file name, patient_id (N,) int32.

pmc: each figure that passes a visual sanity filter (grayscale, aspect 0.7–1.45, not mostly white or black) is
trimmed of uniform borders, centre-cropped to a square and resized to 64×64 grayscale uint8 (Lanczos). 80 % train,
20 % test (default_rng(1)). Two figures found by eye not to be single chest radiographs are excluded (EXCLUDE).
Writes pmc_cxr_subset.npz (images, labels = caption keyword finding 0 none / 1 + first finding, split),
pmc_captions.csv and ATTRIBUTION.csv. sentences.jsonl holds, per article, the body sentences that cite the
figure (and those that mention a chest radiograph), extracted from the same article's PMC XML; the stage-2 text
`report_text` is the caption followed by those sentences (citing ones first, duplicates dropped), whole sentences
only, at most 120 words.
"""

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

NAMES = {"Cardiomegaly": 0, "Effusion": 1, "Infiltration": 2, "Atelectasis": 3, "Pneumothorax": 4, "Mass": 5,
         "Nodule": 5}  # fmt: skip
KEYWORDS = [r"cardiomegaly|enlarged heart|cardiac enlargement|enlarged cardiac", r"effusion",
            r"infiltrat|consolidation", r"atelecta|volume loss|collapse", r"(?<!no )pneumothora",
            r"nodul|mass"]  # fmt: skip
EXCLUDE = {  # found by eye in the montage of the kept figures
    "PMC8775073": "a segmentation pipeline figure (four panels), not one radiograph",
    "PMC4955046": "a skull radiograph (its caption matched 'PA view')",
}
CHANGE = (
    "image downsampled to 64x64, cropped (uniform borders trimmed, centre square), converted to grayscale; "
    "text = figure caption + the article's own sentences citing the figure (whole sentences, <= 120 words)"
)


def nih(csv_path: str, png_dir: str, out: str) -> None:
    rows = {r["Image Index"]: r for r in csv.DictReader(open(csv_path, encoding="utf-8"))}
    imgs, find, names, pids = [], [], [], []
    for f in sorted(Path(png_dir).glob("*.png")):
        y = [0] * 6
        for n in rows[f.name]["Finding Labels"].split("|"):
            if n != "No Finding":
                y[NAMES[n]] = 1
        if sum(y) > 2:
            continue
        with Image.open(f) as im:
            imgs.append(np.asarray(im.convert("L").resize((64, 64), Image.Resampling.BOX), dtype=np.uint8))
        find.append(y)
        names.append(f.name)
        pids.append(int(rows[f.name]["Patient ID"]))
    find = np.array(find, np.uint8)
    lab = np.where(find.any(1), 1 + find.argmax(1), 0).astype(np.int8)
    rng = np.random.default_rng(0)
    split = np.zeros(len(lab), np.uint8)
    for c in np.unique(lab):
        idx = rng.permutation(np.flatnonzero(lab == c))
        split[idx[::4]] = 1
    np.savez_compressed(Path(out) / "nih_cxr_subset.npz", images=np.stack(imgs), labels=lab, findings=find,
                        split=split, image_index=np.array(names), patient_id=np.array(pids, np.int32))  # fmt: skip
    print(len(lab), "images; findings per class", find.sum(0), "no finding", int((lab == 0).sum()),
          "test", int(split.sum()))  # fmt: skip


def trim(a: np.ndarray, tol: float = 4.0) -> np.ndarray:
    """Drop uniform rows and columns at the edges (figure margins)."""
    r = np.flatnonzero(a.std(1) >= tol)
    c = np.flatnonzero(a.std(0) >= tol)
    return a[r[0] : r[-1] + 1, c[0] : c[-1] + 1] if len(r) > 8 and len(c) > 8 else a


def accept(im: Image.Image) -> tuple[bool, str]:
    rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
    h, w = rgb.shape[:2]
    if min(h, w) < 150:
        return False, "small"
    if not 0.7 <= w / h <= 1.45:
        return False, "aspect"
    sat = (np.abs(rgb[..., 0] - rgb[..., 1]) + np.abs(rgb[..., 1] - rgb[..., 2])).mean()
    if sat > 6:
        return False, "colour"
    g = rgb.mean(-1)
    if not 30 <= g.mean() <= 210 or (g > 250).mean() > 0.25:
        return False, "white/black"
    return True, ""


def report_text(caption: str, sent: dict, max_words: int = 120) -> tuple[str, str]:
    """Caption + the article's sentences citing the figure (then those mentioning a chest radiograph), whole
    sentences, ≤ max_words words; and where the extra text came from."""
    cap = caption.strip()
    cap = cap if cap.endswith((".", "!", "?")) else cap + "."  # the caption may lack a full stop
    out, seen, src = [cap], {cap.lower()}, "caption"
    for key, tag in (("cites_figure", "cites figure"), ("mentions_cxr", "mentions chest radiograph")):
        for s in sent.get(key, []):
            s = s.strip()
            if s.lower() in seen or s.lower() in caption.lower():
                continue
            if len(" ".join(out + [s]).split()) > max_words:
                return " ".join(out), src
            out.append(s)
            seen.add(s.lower())
            src = src if tag in src else src + " + " + tag
    return " ".join(out), src


def pmc(records: str, img_dir: str, out: str, sentences: str) -> None:
    recs = [json.loads(line) for line in open(records, encoding="utf-8")]
    sents = {r["pmcid"]: r for r in (json.loads(line) for line in open(sentences, encoding="utf-8"))}
    keep, imgs, reasons = [], [], {}
    for r in recs:
        if r["pmcid"] in EXCLUDE:
            reasons["by eye"] = reasons.get("by eye", 0) + 1
            continue
        with Image.open(Path(img_dir) / r["file"]) as im:
            im.seek(0)
            ok, why = accept(im)
            if not ok:
                reasons[why] = reasons.get(why, 0) + 1
                continue
            a = trim(np.asarray(im.convert("L"), dtype=np.float32))
        s = min(a.shape)
        y0, x0 = (a.shape[0] - s) // 2, (a.shape[1] - s) // 2
        sq = Image.fromarray(a[y0 : y0 + s, x0 : x0 + s].astype(np.uint8))
        imgs.append(np.asarray(sq.resize((64, 64), Image.Resampling.LANCZOS), dtype=np.uint8))
        keep.append(r)
    split = (np.random.default_rng(1).random(len(keep)) < 0.2).astype(np.uint8)
    lab = []
    for r in keep:
        hit = [k for k, p in enumerate(KEYWORDS) if re.search(p, r["caption"].lower())]
        lab.append(1 + hit[0] if hit else 0)
    o = Path(out)
    np.savez_compressed(o / "pmc_cxr_subset.npz", images=np.stack(imgs), labels=np.array(lab, np.int8), split=split)
    with open(o / "pmc_captions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "pmcid", "figure_id", "split", "caption", "report_text", "text_source"])
        for i, (r, s) in enumerate(zip(keep, split)):
            text, src = report_text(r["caption"], sents.get(r["pmcid"], {}))
            w.writerow([i, r["pmcid"], r["fig_id"], "test" if s else "train", r["caption"], text, src])
    with open(o / "ATTRIBUTION.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "pmcid", "figure_id", "figure_label", "authors", "article_title", "journal", "year",
                    "licence", "article_url", "image_url", "fetched", "changes"])  # fmt: skip
        for i, r in enumerate(keep):
            au = r["authors"]
            authors = "; ".join(au[:6]) + ("; et al." if len(au) > 6 else "")
            w.writerow([i, r["pmcid"], r["fig_id"], r["label"], authors, r["title"], r["journal"], r["year"],
                        r["licence"], r["article_url"], r["image_url"], r["fetched"], CHANGE])  # fmt: skip
    print(len(keep), "of", len(recs), "figures kept; rejected", reasons, "; test", int(split.sum()),
          "; caption keyword classes", np.bincount(lab, minlength=7))  # fmt: skip


if __name__ == "__main__":
    {"nih": nih, "pmc": pmc}[sys.argv[1]](*sys.argv[2:])
