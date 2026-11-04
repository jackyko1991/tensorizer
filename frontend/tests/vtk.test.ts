// VTK reader: ascii and base64 (raw and zlib) PolyData, and the .pvd collection. Run: node --test frontend/tests
import assert from "node:assert/strict";
import { test } from "node:test";
import { zlibSync } from "fflate";
import { parsePvd, parseVtp, scalar } from "../src/vtk.ts";

const b64 = (bytes: Uint8Array) => Buffer.from(bytes).toString("base64");
const u32 = (n: number) => new Uint8Array(new Uint32Array([n]).buffer);

function raw(arr: Float32Array | Int32Array): string {
  const bytes = new Uint8Array(arr.buffer);
  return b64(new Uint8Array([...u32(bytes.length), ...bytes]));
}
function zipped(arr: Float32Array | Int32Array): string {
  const bytes = new Uint8Array(arr.buffer);
  const z = zlibSync(bytes);
  const header = new Uint32Array([1, bytes.length, bytes.length, z.length]);
  return b64(new Uint8Array(header.buffer)) + b64(z);
}

function vtp(fmt: "ascii" | "raw" | "zlib"): string {
  const pts = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
  const conn = new Int32Array([0, 1, 2, 0, 2, 3]);
  const offs = new Int32Array([3, 6]);
  const wss = new Float32Array([1, 2, 3, 4]);
  const enc = (a: Float32Array | Int32Array) =>
    fmt === "ascii" ? Array.from(a).join(" ") : fmt === "raw" ? raw(a) : zipped(a);
  const f = fmt === "ascii" ? "ascii" : "binary";
  const comp = fmt === "zlib" ? ' compressor="vtkZLibDataCompressor"' : "";
  return `<?xml version="1.0"?><VTKFile type="PolyData" byte_order="LittleEndian" header_type="UInt32"${comp}>
<PolyData><Piece NumberOfPoints="4" NumberOfPolys="2">
<PointData><DataArray type="Float32" Name="WSS" unit="Pa" format="${f}">${enc(wss)}</DataArray></PointData>
<Points><DataArray type="Float32" NumberOfComponents="3" format="${f}">${enc(pts)}</DataArray></Points>
<Polys><DataArray type="Int32" Name="connectivity" format="${f}">${enc(conn)}</DataArray>
<DataArray type="Int32" Name="offsets" format="${f}">${enc(offs)}</DataArray></Polys>
</Piece></PolyData></VTKFile>`;
}

for (const fmt of ["ascii", "raw", "zlib"] as const) {
  test(`vtp ${fmt}: points, triangles and a named point array with its unit`, () => {
    const d = parseVtp(vtp(fmt));
    assert.equal(d.points.length, 12);
    assert.deepEqual(Array.from(d.polys.connectivity), [0, 1, 2, 0, 2, 3]);
    assert.deepEqual(Array.from(d.polys.offsets), [3, 6]);
    assert.equal(d.pointData[0].name, "WSS");
    assert.equal(d.pointData[0].unit, "Pa");
    assert.deepEqual(Array.from(d.pointData[0].data), [1, 2, 3, 4]);
  });
}

test("pvd: time steps and layers from group / part / file name", () => {
  const entries = parsePvd(`<VTKFile type="Collection"><Collection>
<DataSet timestep="0" group="surface" part="0" file="surface_t000.vtp"/>
<DataSet timestep="0.5" part="particles" file="p.vtp"/>
<DataSet timestep="1" file="streamlines_t002.vtp"/></Collection></VTKFile>`);
  assert.deepEqual(
    entries.map((e) => [e.time, e.layer]),
    [
      [0, "surface"],
      [0.5, "particles"],
      [1, "streamlines"],
    ],
  );
});

test("vector arrays become magnitudes", () => {
  const s = scalar({ name: "u", components: 3, data: new Float32Array([3, 4, 0, 0, 0, 2]) });
  assert.deepEqual(Array.from(s), [5, 2]);
});
