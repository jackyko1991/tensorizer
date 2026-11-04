/** Colormaps shared by the 3D viewers: a few stops each, linearly interpolated (matplotlib's look, small). */

export const CMAPS: Record<string, string[]> = {
  viridis: ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
  magma: ["#000004", "#51127c", "#b73779", "#fc8961", "#fcfdbf"],
  inferno: ["#000004", "#56106e", "#bb3754", "#f98e09", "#fcffa4"],
  plasma: ["#0d0887", "#7e03a8", "#cc4778", "#f89540", "#f0f921"],
  cividis: ["#00224e", "#35456c", "#666970", "#a59c74", "#fee838"],
  coolwarm: ["#3b4cc0", "#8db0fe", "#dddddd", "#f49a7b", "#b40426"],
  turbo: ["#30123b", "#4686fb", "#1be5b5", "#a4fc3c", "#fb8022", "#7a0403"],
};

const rgb = (hex: string) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
const STOPS: Record<string, number[][]> = Object.fromEntries(Object.entries(CMAPS).map(([k, v]) => [k, v.map(rgb)]));

/** Fill `out` (N·3, 0..1) with the colormap of `values` scaled to [lo, hi]. */
export function colorize(values: ArrayLike<number>, lo: number, hi: number, cmap: string, out: Float32Array): void {
  const stops = STOPS[cmap] ?? STOPS.viridis;
  const n = stops.length - 1;
  const span = hi > lo ? hi - lo : 1;
  for (let i = 0; i < values.length; i++) {
    const t = Math.min(1, Math.max(0, (values[i] - lo) / span)) * n;
    const k = Math.min(n - 1, Math.floor(t));
    const f = t - k;
    const a = stops[k];
    const b = stops[k + 1];
    out[3 * i] = a[0] + (b[0] - a[0]) * f;
    out[3 * i + 1] = a[1] + (b[1] - a[1]) * f;
    out[3 * i + 2] = a[2] + (b[2] - a[2]) * f;
  }
}

/** CSS gradient for a legend bar. */
export const gradient = (cmap: string) => `linear-gradient(to right, ${(CMAPS[cmap] ?? CMAPS.viridis).join(",")})`;
