/**
 * A small reader for the VTK files problems write as animated 3D results: XML PolyData (.vtp) and the ParaView
 * collection (.pvd) that lists them per time step and layer. Supports ascii and base64 "binary" DataArrays, raw or
 * zlib-compressed (vtkZLibDataCompressor), with UInt32 or UInt64 headers. Appended data is not supported.
 * Self-contained except for fflate (no DOM: a tiny XML reader), so `node --test frontend/tests` can run it.
 */
import { unzlibSync } from "fflate";

export interface VtkArray {
  name: string;
  components: number;
  data: Float32Array;
  /** Unit from a `unit="…"` attribute on the DataArray, if the writer gave one. */
  unit?: string;
}

export interface PolyData {
  points: Float32Array; // (N·3)
  /** Cells as flat vertex-index lists with offsets (VTK layout), per cell type. */
  verts: { connectivity: Int32Array; offsets: Int32Array };
  lines: { connectivity: Int32Array; offsets: Int32Array };
  polys: { connectivity: Int32Array; offsets: Int32Array };
  pointData: VtkArray[];
}

export interface PvdEntry {
  time: number;
  layer: string;
  file: string;
}

// --- a tiny XML reader (VTK files are simple: elements, attributes, text; no DOM needed, so tests run in Node) ---
interface Node {
  name: string;
  attrs: Record<string, string>;
  children: Node[];
  text: string;
}

function parseXml(xml: string): Node {
  const root: Node = { name: "#root", attrs: {}, children: [], text: "" };
  const stack = [root];
  const re = /<(\/?)([A-Za-z_][\w:.-]*)([^>]*?)(\/?)>|<!--[\s\S]*?-->|<\?[\s\S]*?\?>|([^<]+)/g;
  for (let m; (m = re.exec(xml));) {
    const [, close, name, rest, self, text] = m;
    const top = stack[stack.length - 1];
    if (text !== undefined) {
      top.text += text;
      continue;
    }
    if (!name) continue; // comment or <?xml?>
    if (close) {
      stack.pop();
      continue;
    }
    const attrs: Record<string, string> = {};
    for (const a of rest.matchAll(/([\w:.-]+)\s*=\s*"([^"]*)"/g)) attrs[a[1]] = a[2];
    const node: Node = { name, attrs, children: [], text: "" };
    top.children.push(node);
    if (!self) stack.push(node);
  }
  return root;
}

function all(node: Node, name: string): Node[] {
  const out: Node[] = [];
  const walk = (n: Node) => {
    for (const c of n.children) {
      if (c.name === name) out.push(c);
      walk(c);
    }
  };
  walk(node);
  return out;
}
const first = (node: Node | undefined, name: string) => (node ? all(node, name)[0] : undefined);

const TYPED: Record<string, (b: ArrayBuffer) => ArrayLike<number>> = {
  Float32: (b) => new Float32Array(b),
  Float64: (b) => new Float64Array(b),
  Int8: (b) => new Int8Array(b),
  UInt8: (b) => new Uint8Array(b),
  Int16: (b) => new Int16Array(b),
  UInt16: (b) => new Uint16Array(b),
  Int32: (b) => new Int32Array(b),
  UInt32: (b) => new Uint32Array(b),
  Int64: (b) => new BigInt64Array(b) as unknown as ArrayLike<number>,
  UInt64: (b) => new BigUint64Array(b) as unknown as ArrayLike<number>,
};

function base64(text: string): Uint8Array {
  const clean = text.replace(/\s+/g, "");
  if (typeof atob === "function") return Uint8Array.from(atob(clean), (c) => c.charCodeAt(0));
  return new Uint8Array(Buffer.from(clean, "base64")); // Node (tests)
}

/** Read `count` header integers of the given byte size from the start of base64 text, returning them and the bytes. */
function readHeader(bytes: Uint8Array, count: number, size: 4 | 8): number[] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Array.from({ length: count }, (_, i) =>
    size === 4 ? view.getUint32(i * 4, true) : Number(view.getBigUint64(i * 8, true)),
  );
}

/** The raw bytes of one binary DataArray (base64; zlib blocks when compressed). */
function binaryBytes(text: string, compressed: boolean, headerSize: 4 | 8): Uint8Array {
  if (!compressed) {
    const all = base64(text);
    const [n] = readHeader(all, 1, headerSize);
    return all.subarray(headerSize, headerSize + n);
  }
  // compressed: the header (blocks, block size, last size, sizes…) is its own base64 chunk, then the blocks
  const clean = text.replace(/\s+/g, "");
  const firstQuad = base64(clean.slice(0, Math.ceil((3 * headerSize) / 3) * 4 + 8));
  const blocks = readHeader(firstQuad, 1, headerSize)[0];
  const headerBytes = (3 + blocks) * headerSize;
  const headerChars = Math.ceil(headerBytes / 3) * 4;
  const header = readHeader(base64(clean.slice(0, headerChars)), 3 + blocks, headerSize);
  const sizes = header.slice(3);
  const body = base64(clean.slice(headerChars));
  const parts: Uint8Array[] = [];
  let at = 0;
  for (const size of sizes) {
    parts.push(unzlibSync(body.subarray(at, at + size)));
    at += size;
  }
  const total = parts.reduce((n, p) => n + p.length, 0);
  const out = new Uint8Array(total);
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

function readArray(el: Node, compressed: boolean, headerSize: 4 | 8): number[] | ArrayLike<number> {
  const format = el.attrs.format ?? "ascii";
  const type = el.attrs.type ?? "Float32";
  const text = el.text;
  if (format === "ascii") return text.trim() ? text.trim().split(/\s+/).map(Number) : [];
  if (format !== "binary") throw new Error(`VTK: DataArray format "${format}" is not supported (use ascii or binary)`);
  const bytes = binaryBytes(text, compressed, headerSize);
  const copy = bytes.slice().buffer; // aligned copy
  const make = TYPED[type];
  if (!make) throw new Error(`VTK: DataArray type ${type} is not supported`);
  return make(copy);
}

const toF32 = (a: ArrayLike<number>) => Float32Array.from(a as ArrayLike<number>, (v) => Number(v));
const toI32 = (a: ArrayLike<number>) => Int32Array.from(a as ArrayLike<number>, (v) => Number(v));

/** Parse a .vtp (XML PolyData) document. All pieces are merged into one. */
export function parseVtp(xml: string): PolyData {
  const root = first(parseXml(xml), "VTKFile");
  if (!root || root.attrs.type !== "PolyData") throw new Error("VTK: not a PolyData .vtp file");
  const compressed = !!root.attrs.compressor;
  const headerSize = root.attrs.header_type === "UInt64" ? 8 : 4;
  const pieces = all(root, "Piece");
  const pts: number[] = [];
  const cells = { verts: [[], []] as number[][], lines: [[], []] as number[][], polys: [[], []] as number[][] };
  const arrays = new Map<string, { components: number; unit?: string; values: number[] }>();
  for (const piece of pieces) {
    const base = pts.length / 3;
    const pointsEl = first(first(piece, "Points"), "DataArray");
    if (pointsEl) pts.push(...Array.from(readArray(pointsEl, compressed, headerSize) as ArrayLike<number>, Number));
    for (const kind of ["Verts", "Lines", "Polys"] as const) {
      const el = first(piece, kind);
      if (!el) continue;
      const das = all(el, "DataArray");
      const conn = das.find((d) => d.attrs.Name === "connectivity");
      const offs = das.find((d) => d.attrs.Name === "offsets");
      if (!conn || !offs) continue;
      const key = kind.toLowerCase() as "verts" | "lines" | "polys";
      const [c, o] = cells[key];
      const shift = c.length;
      c.push(...Array.from(readArray(conn, compressed, headerSize) as ArrayLike<number>, (v) => Number(v) + base));
      o.push(...Array.from(readArray(offs, compressed, headerSize) as ArrayLike<number>, (v) => Number(v) + shift));
    }
    const pd = first(piece, "PointData");
    for (const da of pd ? all(pd, "DataArray") : []) {
      const name = da.attrs.Name ?? "array";
      const entry = arrays.get(name) ?? {
        components: Number(da.attrs.NumberOfComponents ?? 1),
        unit: da.attrs.unit,
        values: [],
      };
      entry.values.push(...Array.from(readArray(da, compressed, headerSize) as ArrayLike<number>, Number));
      arrays.set(name, entry);
    }
  }
  const pair = ([c, o]: number[][]) => ({ connectivity: toI32(c), offsets: toI32(o) });
  return {
    points: toF32(pts),
    verts: pair(cells.verts),
    lines: pair(cells.lines),
    polys: pair(cells.polys),
    pointData: [...arrays].map(([name, a]) => ({
      name,
      components: a.components,
      unit: a.unit,
      data: toF32(a.values),
    })),
  };
}

/** Parse a ParaView .pvd collection: one entry per DataSet (time, layer from group/part/file name, file). */
export function parsePvd(xml: string): PvdEntry[] {
  return all(parseXml(xml), "DataSet").map((ds) => {
    const a = ds.attrs;
    const file = a.file ?? "";
    const layer =
      a.name ||
      a.group ||
      (a.part && !/^\d+$/.test(a.part) ? a.part : "") ||
      file
        .split("/")
        .pop()!
        .replace(/(_t?\d+)?\.vtp$/i, "");
    return { time: Number(a.timestep ?? 0), layer: layer || "layer", file };
  });
}

/** Scalar magnitude of a point array (vectors → length), as one value per point. */
export function scalar(a: VtkArray): Float32Array {
  if (a.components === 1) return a.data;
  const n = a.data.length / a.components;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = 0; k < a.components; k++) s += a.data[i * a.components + k] ** 2;
    out[i] = Math.sqrt(s);
  }
  return out;
}
