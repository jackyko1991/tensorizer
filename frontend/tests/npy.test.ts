// NumPy .npy/.npz reader for image datasets. Run: node --test frontend/tests
import assert from "node:assert/strict";
import { test } from "node:test";
import { zipSync } from "fflate";
import { labelKey, parseImageNpz, parseNpy, rgba, toInt32 } from "../src/npy.ts";

/** A .npy file as numpy writes it: magic, version, little-endian header length, padded dict header, data. */
function npy(descr: string, shape: number[], data: Uint8Array, version = 1, fortran = false): Uint8Array {
  const shapeText = shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(", ")})`;
  let dict = `{'descr': '${descr}', 'fortran_order': ${fortran ? "True" : "False"}, 'shape': ${shapeText}, }`;
  const pre = version === 1 ? 10 : 12;
  while ((pre + dict.length + 1) % 64) dict += " ";
  dict += "\n";
  const head = new Uint8Array(pre);
  head.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, version, 0]);
  const v = new DataView(head.buffer);
  if (version === 1) v.setUint16(8, dict.length, true);
  else v.setUint32(8, dict.length, true);
  return new Uint8Array([...head, ...new TextEncoder().encode(dict), ...data]);
}

const bytes = (a: ArrayBufferView) => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);

test("npy v1 and v2 headers: shape, dtype and data", () => {
  const data = new Uint8Array([1, 2, 3, 4, 5, 6]);
  for (const version of [1, 2]) {
    const a = parseNpy(npy("|u1", [1, 2, 3], data, version));
    assert.equal(a.descr, "|u1");
    assert.deepEqual(a.shape, [1, 2, 3]);
    assert.deepEqual([...a.data], [1, 2, 3, 4, 5, 6]);
  }
});

test("npy: other versions, Fortran order, bad magic and truncation are rejected with a message", () => {
  const data = new Uint8Array(4);
  const v3 = npy("|u1", [4], data, 2);
  v3[6] = 3;
  assert.throws(() => parseNpy(v3), /version 3\.0/);
  assert.throws(() => parseNpy(npy("|u1", [2, 2], data, 1, true)), /Fortran/);
  assert.throws(() => parseNpy(new Uint8Array(20)), /magic/);
  assert.throws(() => parseNpy(npy("|u1", [8], data)), /Truncated/);
});

test("labels: little-endian integers of any width become Int32Array; big-endian is rejected", () => {
  const i64 = bytes(new BigInt64Array([0n, 9n, -1n]));
  assert.deepEqual([...toInt32(parseNpy(npy("<i8", [3], i64)))], [0, 9, -1]);
  assert.deepEqual([...toInt32(parseNpy(npy("<u2", [2], bytes(new Uint16Array([7, 65535])))))], [7, 65535]);
  assert.throws(() => toInt32(parseNpy(npy(">i4", [1], new Uint8Array(4)))), /little-endian/);
});

test("npz: images + labels, deflated, and x_train/y_train splits", () => {
  const imgs = new Uint8Array(3 * 2 * 2).map((_, i) => i * 10);
  const npz = zipSync(
    {
      "images.npy": npy("|u1", [3, 2, 2], imgs),
      "labels.npy": npy("<i8", [3], bytes(new BigInt64Array([4n, 2n, 4n]))),
      "x_test.npy": npy("|u1", [2, 1, 1], new Uint8Array([5, 6])),
      "y_test.npy": npy("|u1", [2], new Uint8Array([1, 0])),
      "idx_test.npy": npy("<i4", [2], bytes(new Int32Array([10, 20]))),
    },
    { level: 6 },
  );
  const sets = parseImageNpz(npz);
  assert.deepEqual(
    sets.map((s) => s.name),
    ["images", "x_test"],
  );
  const [a, b] = sets;
  assert.equal(a.count, 3);
  assert.equal(a.height, 2);
  assert.equal(a.width, 2);
  assert.equal(a.channels, 1);
  assert.equal(a.labelsName, "labels");
  assert.deepEqual([...a.labels!], [4, 2, 4]);
  assert.deepEqual([...a.images.subarray(4, 8)], [40, 50, 60, 70]);
  assert.deepEqual([...b.labels!], [1, 0]);
  // grey → RGBA, image 1
  assert.deepEqual([...rgba(a, 1).subarray(0, 8)], [40, 40, 40, 255, 50, 50, 50, 255]);
});

test("npz: images without matching labels, RGB images, and archives with no images", () => {
  const rgb = new Uint8Array([1, 2, 3, 4, 5, 6]);
  const [s] = parseImageNpz(
    zipSync({
      "imgs.npy": npy("|u1", [1, 1, 2, 3], rgb),
      "labels.npy": npy("|u1", [5], new Uint8Array(5)), // wrong length: not paired
    }),
  );
  assert.equal(s.channels, 3);
  assert.equal(s.labels, null);
  assert.deepEqual([...rgba(s, 0)], [1, 2, 3, 255, 4, 5, 6, 255]);
  assert.throws(() => parseImageNpz(zipSync({ "u.npy": npy("<f8", [1], new Uint8Array(8)) })), /No image array/);
  assert.throws(() => parseImageNpz(new Uint8Array([1, 2, 3])), /npz/);
});

test("label pairing by name", () => {
  assert.equal(labelKey("x_train", ["x_train", "y_train"]), "y_train");
  assert.equal(labelKey("X", ["X", "y"]), "y");
  assert.equal(labelKey("train_images", ["train_images", "train_labels"]), "train_labels");
  assert.equal(labelKey("images", ["images"]), null);
});
