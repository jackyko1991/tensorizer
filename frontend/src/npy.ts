/**
 * NumPy .npy / .npz reader for image datasets: uint8 image stacks (N,H,W) or (N,H,W,1|3) and their integer labels.
 *
 * No DOM (only fflate), so `node --test frontend/tests` can run it.
 */
import { unzipSync } from "fflate";

export interface NpyArray {
  /** NumPy dtype string, e.g. "|u1", "<i8". */
  descr: string;
  shape: number[];
  /** The raw array bytes (C order), a view into the file. */
  data: Uint8Array;
}

export interface ImageSet {
  /** The image array's name in the archive, e.g. "images" or "x_train". */
  name: string;
  /** Pixels, N·H·W·C bytes in C order. */
  images: Uint8Array;
  count: number;
  height: number;
  width: number;
  /** 1 (grey) or 3 (RGB). */
  channels: number;
  /** The paired labels array's name, or null. */
  labelsName: string | null;
  labels: Int32Array | null;
}

const MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]; // \x93NUMPY

/** One .npy file → its header and a view of its data. Only format versions 1 and 2 and C order are read. */
export function parseNpy(bytes: Uint8Array): NpyArray {
  if (bytes.length < 10 || MAGIC.some((b, i) => bytes[i] !== b)) throw new Error("Not a .npy file (bad magic).");
  const major = bytes[6];
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let headerLen: number;
  let start: number;
  if (major === 1) {
    headerLen = view.getUint16(8, true);
    start = 10;
  } else if (major === 2) {
    if (bytes.length < 12) throw new Error("Truncated .npy header.");
    headerLen = view.getUint32(8, true);
    start = 12;
  } else throw new Error(`Unsupported .npy format version ${major}.${bytes[7]} (only 1.0 and 2.0 are read).`);
  if (start + headerLen > bytes.length) throw new Error("Truncated .npy header.");
  let header = "";
  for (let i = start; i < start + headerLen; i++) header += String.fromCharCode(bytes[i]);
  const descr = /'descr'\s*:\s*'([^']*)'/.exec(header)?.[1];
  const fortran = /'fortran_order'\s*:\s*(True|False)/.exec(header)?.[1];
  const shapeText = /'shape'\s*:\s*\(([^)]*)\)/.exec(header)?.[1];
  if (descr === undefined || fortran === undefined || shapeText === undefined)
    throw new Error("Unreadable .npy header.");
  if (fortran === "True") throw new Error("Fortran-ordered arrays are not supported (save in C order).");
  const shape = shapeText
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number);
  if (shape.some((n) => !Number.isInteger(n) || n < 0)) throw new Error("Unreadable .npy shape.");
  const offset = start + headerLen;
  const size = itemSize(descr) * shape.reduce((a, b) => a * b, 1);
  if (offset + size > bytes.length) throw new Error("Truncated .npy data.");
  return { descr, shape, data: bytes.subarray(offset, offset + size) };
}

/** Bytes per element of an integer dtype (1, 2, 4, 8); 0 when it is not a plain integer type. */
function intSize(descr: string): number {
  const m = /^([<|=])([iu])([1248])$/.exec(descr);
  return m ? Number(m[3]) : 0;
}

function itemSize(descr: string): number {
  const m = /^[<>|=][a-zA-Z](\d+)$/.exec(descr);
  if (!m) throw new Error(`Unsupported dtype ${descr}.`);
  return Number(m[1]);
}

/** An integer array → Int32Array (little-endian only). */
export function toInt32(a: NpyArray): Int32Array {
  const size = intSize(a.descr);
  if (!size) throw new Error(`Labels must be little-endian integers, got ${a.descr}.`);
  const n = a.shape.reduce((x, y) => x * y, 1);
  const v = new DataView(a.data.buffer, a.data.byteOffset, a.data.byteLength);
  const out = new Int32Array(n);
  const signed = a.descr[1] === "i";
  for (let i = 0; i < n; i++) {
    const o = i * size;
    out[i] =
      size === 1
        ? signed ? v.getInt8(o) : v.getUint8(o)
        : size === 2
          ? signed ? v.getInt16(o, true) : v.getUint16(o, true)
          : size === 4
            ? signed ? v.getInt32(o, true) : v.getUint32(o, true)
            : Number(signed ? v.getBigInt64(o, true) : v.getBigUint64(o, true));
  }
  return out;
}

/** An .npz archive (zip, stored or deflated) → its arrays by name (without the .npy suffix). */
export function parseNpz(buf: Uint8Array): Record<string, NpyArray> {
  let files: Record<string, Uint8Array>;
  try {
    files = unzipSync(buf, { filter: (f) => f.name.endsWith(".npy") });
  } catch (e) {
    throw new Error(`Not a readable .npz archive (${e instanceof Error ? e.message : String(e)}).`);
  }
  const out: Record<string, NpyArray> = {};
  for (const [name, bytes] of Object.entries(files)) out[name.slice(0, -4)] = parseNpy(bytes);
  return out;
}

const isImages = (a: NpyArray) =>
  (a.descr === "|u1" || a.descr === "<u1" || a.descr === "=u1") &&
  (a.shape.length === 3 || (a.shape.length === 4 && (a.shape[3] === 1 || a.shape[3] === 3)));

/** The labels array paired with an image array by name: images/labels, x/y, x_train/y_train, <p>_images/<p>_labels. */
export function labelKey(imageKey: string, keys: string[]): string | null {
  const low = imageKey.toLowerCase();
  const candidates: string[] = [];
  if (low.startsWith("x")) candidates.push("y" + imageKey.slice(1), "Y" + imageKey.slice(1));
  for (const word of ["images", "image", "imgs", "img"]) {
    const i = low.indexOf(word);
    if (i >= 0)
      for (const rep of ["labels", "label", "targets", "y"])
        candidates.push(imageKey.slice(0, i) + rep + imageKey.slice(i + word.length));
  }
  return candidates.find((c) => keys.includes(c)) ?? null;
}

/** An image-dataset .npz → its image sets (one per uint8 image array), each with its paired labels if any. */
export function parseImageNpz(buf: Uint8Array): ImageSet[] {
  const arrays = parseNpz(buf);
  const keys = Object.keys(arrays);
  const sets: ImageSet[] = [];
  for (const k of keys) {
    const a = arrays[k];
    if (!isImages(a)) continue;
    const [count, height, width] = a.shape;
    const lk = labelKey(k, keys);
    let labels: Int32Array | null = null;
    if (lk) {
      const l = arrays[lk];
      if (l.shape.length === 1 && l.shape[0] === count && intSize(l.descr)) labels = toInt32(l);
    }
    sets.push({
      name: k,
      images: a.data,
      count,
      height,
      width,
      channels: a.shape[3] ?? 1,
      labelsName: labels ? lk : null,
      labels,
    });
  }
  if (!sets.length)
    throw new Error("No image array in this .npz (expected uint8 with shape (N, H, W) or (N, H, W, 1|3)).");
  return sets;
}

/** Image i of a set as RGBA pixels (for ImageData). */
export function rgba(set: ImageSet, i: number): Uint8ClampedArray<ArrayBuffer> {
  const n = set.height * set.width;
  const c = set.channels;
  const src = set.images.subarray(i * n * c, (i + 1) * n * c);
  const out = new Uint8ClampedArray(n * 4);
  for (let p = 0; p < n; p++) {
    const r = src[p * c];
    out[p * 4] = r;
    out[p * 4 + 1] = c === 3 ? src[p * c + 1] : r;
    out[p * 4 + 2] = c === 3 ? src[p * c + 2] : r;
    out[p * 4 + 3] = 255;
  }
  return out;
}
