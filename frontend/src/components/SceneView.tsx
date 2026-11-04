/** An animated, layered 3D result from VTK files: a ParaView .pvd collection (or one .vtp) whose layers (e.g. surface,
 *  streamlines, particles) can be switched on and off, coloured by any of their point arrays, and played through
 *  their time steps. Rendering is three.js, loaded only when a scene is opened; nothing here is specific to one
 *  problem. */
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { colorize, gradient } from "../colormaps";
import { useLang } from "../i18n";
import { parsePvd, parseVtp, scalar, type PolyData } from "../vtk";

interface Layer {
  name: string;
  /** One file per time step (a static layer has one file for all steps). */
  files: Map<number, string>;
  /** Solid colour (from the .pvd) when there is nothing to colour by. */
  color?: string;
}

interface Loaded {
  times: number[];
  layers: Layer[];
  data: Map<string, PolyData>; // file → parsed
  /** Range of every (layer, array) over all time steps, so colours stay fixed while it plays. */
  ranges: Map<string, [number, number]>;
  units: Map<string, string | undefined>;
  span: number;
}

const CMAP_FOR = (name: string) =>
  /age|time/i.test(name) ? "plasma" : /cp|pressure|delta|Δ|residual/i.test(name) ? "coolwarm" : "viridis";

/** The file a layer shows at time t: its own step, else the latest earlier one (static layers: the only one). */
function fileAt(layer: Layer, t: number): string | undefined {
  if (layer.files.has(t)) return layer.files.get(t);
  let best: number | undefined;
  for (const k of layer.files.keys()) if (k <= t && (best === undefined || k > best)) best = k;
  return layer.files.get(best ?? [...layer.files.keys()][0]);
}

export default function SceneView({ url, fill = false }: { url: string; fill?: boolean }) {
  const { t } = useLang();
  const host = useRef<HTMLDivElement>(null);
  const [scene, setScene] = useState<Loaded | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [visible, setVisible] = useState<Record<string, boolean>>({});
  const [colorBy, setColorBy] = useState<Record<string, string>>({});
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [fps, setFps] = useState(6);
  const [dark, setDark] = useState(false);
  // surfaces see-through when the scene has flow inside them (streamlines, particles), so it shows
  const [surfaceOpacity, setSurfaceOpacity] = useState(1);
  const [expanded, setExpanded] = useState(false);
  const [panelOpen, setPanelOpen] = useState(true);
  const view = useRef<{ group: THREE.Group; home: () => void; renderer: THREE.WebGLRenderer } | null>(null);

  // 1. load the collection and every file it lists (budgeted to ~10 MB by the problems)
  useEffect(() => {
    let live = true;
    (async () => {
      try {
        const base = url.split("?")[0];
        const dir = base.slice(0, base.lastIndexOf("/") + 1);
        const query = url.includes("?") ? url.slice(url.indexOf("?")) : "";
        const text = await (await fetch(url)).text();
        const entries = base.endsWith(".pvd")
          ? parsePvd(text)
          : [{ time: 0, layer: "mesh", file: base.slice(dir.length) }];
        const files = [...new Set(entries.map((e) => e.file))];
        const data = new Map<string, PolyData>();
        let done = 0;
        await Promise.all(
          files.map(async (f) => {
            const body = base.endsWith(".pvd") ? await (await fetch(dir + f + query)).text() : text;
            data.set(f, parseVtp(body));
            if (live) setProgress(++done / files.length);
          }),
        );
        const layers = new Map<string, Layer>();
        for (const e of entries) {
          if (!layers.has(e.layer)) layers.set(e.layer, { name: e.layer, files: new Map(), color: e.color });
          layers.get(e.layer)!.files.set(e.time, e.file);
        }
        const ranges = new Map<string, [number, number]>();
        const units = new Map<string, string | undefined>();
        const box = new THREE.Box3();
        for (const L of layers.values())
          for (const f of L.files.values()) {
            const d = data.get(f)!;
            for (let i = 0; i < d.points.length; i += 3)
              box.expandByPoint(new THREE.Vector3(d.points[i], d.points[i + 1], d.points[i + 2]));
            for (const a of d.pointData) {
              const key = `${L.name}/${a.name}`;
              const v = scalar(a);
              let [lo, hi] = ranges.get(key) ?? [Infinity, -Infinity];
              for (let i = 0; i < v.length; i++) {
                if (v[i] < lo) lo = v[i];
                if (v[i] > hi) hi = v[i];
              }
              ranges.set(key, [lo, hi]);
              units.set(key, a.unit);
            }
          }
        // an array that is empty in every step (a layer that only appears later) has no range yet
        for (const [k, [lo, hi]] of ranges) if (!(lo <= hi)) ranges.set(k, [0, 1]);
        if (!live) return;
        const loaded: Loaded = {
          times: [...new Set(entries.map((e) => e.time))].sort((a, b) => a - b),
          layers: [...layers.values()],
          data,
          ranges,
          units,
          span: box.getSize(new THREE.Vector3()).length() || 1,
        };
        setVisible(Object.fromEntries(loaded.layers.map((L) => [L.name, true])));
        setColorBy(
          Object.fromEntries(
            loaded.layers.map((L) => [L.name, data.get(fileAt(L, loaded.times[0])!)?.pointData[0]?.name ?? ""]),
          ),
        );
        const hasFlow = [...data.values()].some((d) => d.lines.offsets.length || d.verts.offsets.length);
        setSurfaceOpacity(hasFlow ? 0.35 : 1);
        // the layer panel starts open only when there is room for it (a big or expanded view)
        setPanelOpen((host.current?.clientHeight ?? 0) > 520);
        setScene(loaded);
      } catch (e) {
        if (live) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      live = false;
    };
  }, [url]);

  // 2. the three.js view (once per scene)
  useEffect(() => {
    const el = host.current;
    if (!el || !scene) return;
    let stop = false;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);
    const world = new THREE.Scene();
    world.add(new THREE.HemisphereLight(0xffffff, 0x8a8f99, 1.2));
    const sun = new THREE.DirectionalLight(0xffffff, 1.2);
    sun.position.set(2, 3, 2);
    world.add(sun);
    const group = new THREE.Group();
    group.rotation.x = -Math.PI / 2; // VTK z-up → three.js y-up
    world.add(group);
    const s = scene.span;
    const camera = new THREE.PerspectiveCamera(35, 1, s / 1000, s * 100);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    const box = new THREE.Box3();
    for (const d of scene.data.values())
      for (let i = 0; i < d.points.length; i += 3)
        box.expandByPoint(new THREE.Vector3(d.points[i], d.points[i + 2], -d.points[i + 1]));
    const centre = box.getCenter(new THREE.Vector3());
    const home = () => {
      const back = 1 / Math.min(1, camera.aspect);
      camera.position.set(centre.x - s * 0.9 * back, centre.y + s * 0.55 * back, centre.z + s * 0.9 * back);
      controls.target.copy(centre);
      controls.update();
    };
    const resize = () => {
      renderer.setSize(el.clientWidth, el.clientHeight, false);
      renderer.domElement.style.width = "100%";
      renderer.domElement.style.height = "100%";
      camera.aspect = el.clientWidth / Math.max(el.clientHeight, 1);
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(el);
    resize();
    home();
    view.current = { group, home, renderer };
    const tick = () => {
      if (stop) return;
      controls.update();
      renderer.render(world, camera);
      requestAnimationFrame(tick);
    };
    tick();
    return () => {
      stop = true;
      view.current = null;
      observer.disconnect();
      controls.dispose();
      group.traverse((o) => {
        if (o instanceof THREE.Mesh || o instanceof THREE.LineSegments || o instanceof THREE.Points) {
          o.geometry.dispose();
          (o.material as THREE.Material).dispose();
        }
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [scene]);

  // 3. rebuild the visible layers for the current step / colouring
  useEffect(() => {
    const v = view.current;
    if (!v || !scene) return;
    for (const child of [...v.group.children]) {
      v.group.remove(child);
      if (child instanceof THREE.Mesh || child instanceof THREE.LineSegments || child instanceof THREE.Points) {
        child.geometry.dispose();
        (child.material as THREE.Material).dispose();
      }
    }
    const time = scene.times[Math.min(step, scene.times.length - 1)];
    for (const L of scene.layers) {
      if (!visible[L.name]) continue;
      const d = scene.data.get(fileAt(L, time) ?? "");
      if (!d) continue;
      const position = new THREE.BufferAttribute(d.points, 3);
      const arr = d.pointData.find((a) => a.name === colorBy[L.name]);
      const colors = new Float32Array(d.points.length);
      const range = arr ? scene.ranges.get(`${L.name}/${arr.name}`) : undefined;
      if (arr && range && range[1] > range[0]) colorize(scalar(arr), range[0], range[1], CMAP_FOR(arr.name), colors);
      else if (L.color) {
        // nothing to colour by, and the .pvd names a colour (inlet / outlet caps…)
        const c = new THREE.Color(L.color);
        for (let i = 0; i < colors.length; i += 3) colors.set([c.r, c.g, c.b], i);
      } else colors.fill(0.6); // nothing to colour by, or a constant array: a plain neutral grey
      const color = new THREE.BufferAttribute(colors, 3);
      if (d.polys.offsets.length) {
        const idx: number[] = [];
        let start = 0;
        for (const end of d.polys.offsets) {
          for (let k = start + 1; k < end - 1; k++)
            idx.push(d.polys.connectivity[start], d.polys.connectivity[k], d.polys.connectivity[k + 1]);
          start = end;
        }
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", position);
        g.setAttribute("color", color);
        g.setIndex(idx);
        g.computeVertexNormals();
        const first =
          L.name === scene.layers.find((x) => scene.data.get(fileAt(x, time) ?? "")?.polys.offsets.length)?.name;
        // later surfaces (wakes) see-through; solid-colour markers (inlet / outlet glyphs) always opaque
        const alpha = L.color ? 1 : first ? surfaceOpacity : Math.min(surfaceOpacity, 0.45);
        const opaque = alpha >= 1;
        v.group.add(
          new THREE.Mesh(
            g,
            new THREE.MeshStandardMaterial({
              vertexColors: true,
              side: THREE.DoubleSide,
              roughness: 0.6,
              transparent: !opaque,
              opacity: alpha,
              depthWrite: opaque,
            }),
          ),
        );
      }
      if (d.lines.offsets.length) {
        const seg: number[] = [];
        let start = 0;
        for (const end of d.lines.offsets) {
          for (let k = start; k < end - 1; k++) seg.push(d.lines.connectivity[k], d.lines.connectivity[k + 1]);
          start = end;
        }
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", position);
        g.setAttribute("color", color);
        g.setIndex(seg);
        v.group.add(
          new THREE.LineSegments(
            g,
            new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.6 }),
          ),
        );
      }
      if (d.verts.offsets.length) {
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", position);
        g.setAttribute("color", color);
        v.group.add(new THREE.Points(g, new THREE.PointsMaterial({ vertexColors: true, size: scene.span / 250 })));
      }
    }
    v.renderer.setClearColor(dark ? 0x111827 : 0x000000, dark ? 1 : 0);
  }, [scene, step, visible, colorBy, dark, surfaceOpacity]);

  // 4. play
  useEffect(() => {
    if (!playing || !scene || scene.times.length < 2) return;
    const id = window.setInterval(() => setStep((s) => (s + 1) % scene.times.length), 1000 / fps);
    return () => window.clearInterval(id);
  }, [playing, fps, scene]);
  useEffect(() => {
    if (!expanded) return;
    const key = (e: KeyboardEvent) => e.key === "Escape" && setExpanded(false);
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [expanded]);

  const legends = useMemo(() => {
    if (!scene) return [];
    return scene.layers
      .filter((L) => {
        const r = scene.ranges.get(`${L.name}/${colorBy[L.name]}`);
        return visible[L.name] && colorBy[L.name] && r && r[1] > r[0]; // no colour bar for a constant array
      })
      .map((L) => {
        const key = `${L.name}/${colorBy[L.name]}`;
        const [lo, hi] = scene.ranges.get(key) ?? [0, 1];
        return {
          layer: L.name,
          array: colorBy[L.name],
          unit: scene.units.get(key),
          lo,
          hi,
          cmap: CMAP_FOR(colorBy[L.name]),
        };
      });
  }, [scene, visible, colorBy]);

  const arraysOf = (L: Layer) => {
    if (!scene) return [];
    const d = scene.data.get(fileAt(L, scene.times[0]) ?? "");
    return d?.pointData.map((a) => a.name) ?? [];
  };

  return (
    <div
      className={`flex flex-col ${dark ? "bg-gray-900 text-gray-100" : "bg-white"} ${
        expanded ? "fixed inset-3 z-50 rounded border border-gray-200 shadow-2xl" : fill ? "h-full" : "h-96"
      }`}
      data-testid="scene-view"
    >
      <div className="relative min-h-0 flex-1">
        <div ref={host} className="h-full w-full" />
        {!scene && !error && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-xs text-gray-500">
            <span className="h-8 w-8 animate-spin rounded-full border-2 border-gray-200 border-t-blue-600" />
            {Math.round(progress * 100)}%
          </div>
        )}
        {error && <p className="absolute inset-0 p-4 text-xs text-red-600">{error}</p>}
        {scene && (
          <>
            {/* one compact panel: per layer its switch, colour-by and colour bar (legend), so nothing overlaps */}
            <div
              className={`absolute top-1 left-1 max-h-[calc(100%-0.5rem)] ${panelOpen ? "w-60" : "w-auto"} space-y-2 overflow-y-auto rounded-md border border-gray-200 bg-white/90 p-2 text-[13px] text-gray-800 shadow-sm`}
              data-testid="scene-layers"
            >
              <button
                onClick={() => setPanelOpen((o) => !o)}
                aria-expanded={panelOpen}
                data-testid="scene-layers-toggle"
                className="flex w-full items-center justify-between rounded px-0.5 text-xs font-semibold text-gray-600 hover:bg-gray-100"
              >
                <span>
                  {t("scene.layers")} ({scene.layers.filter((L) => visible[L.name]).length}/{scene.layers.length})
                </span>
                <span aria-hidden>{panelOpen ? "▾" : "▸"}</span>
              </button>
              {panelOpen && scene.layers.map((L) => {
                const legend = legends.find((l) => l.layer === L.name);
                return (
                  <div key={L.name}>
                    <div className="flex items-center gap-1">
                      <label className="flex min-w-0 flex-1 cursor-pointer items-center gap-1">
                        <input
                          type="checkbox"
                          checked={!!visible[L.name]}
                          onChange={() => setVisible((v) => ({ ...v, [L.name]: !v[L.name] }))}
                          data-testid={`scene-layer-${L.name}`}
                        />
                        {L.color && (
                          <span
                            className="h-3 w-3 shrink-0 rounded-sm border border-black/20"
                            style={{ background: L.color }}
                          />
                        )}
                        <span className="truncate font-medium">{L.name}</span>
                      </label>
                      {arraysOf(L).length > 1 && (
                        <select
                          value={colorBy[L.name]}
                          onChange={(e) => setColorBy((c) => ({ ...c, [L.name]: e.target.value }))}
                          className="max-w-24 rounded border border-gray-200 bg-white px-0.5 text-xs"
                          aria-label={t("scene.colorBy")}
                        >
                          {arraysOf(L).map((a) => (
                            <option key={a}>{a}</option>
                          ))}
                        </select>
                      )}
                    </div>
                    {legend && (
                      <div className="mt-0.5 pl-5 text-xs text-gray-700" data-testid="scene-legend">
                        <p className="leading-tight">
                          {legend.array}
                          {legend.unit ? ` (${legend.unit})` : ""}
                        </p>
                        <div className="my-0.5 h-2 rounded-sm" style={{ background: gradient(legend.cmap) }} />
                        <p className="flex justify-between gap-1 tabular-nums">
                          <span>{legend.lo.toPrecision(3)}</span>
                          <span>{legend.hi.toPrecision(3)}</span>
                        </p>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
            <div className="absolute top-1 right-1 flex items-center gap-0.5 rounded-md border border-gray-200 bg-white/90 p-0.5 text-xs text-gray-500 shadow-sm">
              <input
                type="range"
                min={0.1}
                max={1}
                step={0.05}
                value={surfaceOpacity}
                onChange={(e) => setSurfaceOpacity(Number(e.target.value))}
                title={`${t("mesh.opacity")}: ${Math.round(surfaceOpacity * 100)}%`}
                aria-label={t("mesh.opacity")}
                data-testid="scene-opacity"
                className="mx-1 h-1 w-14 cursor-pointer accent-blue-600"
              />
              <button
                onClick={() => setDark((d) => !d)}
                title={t("mesh.dark")}
                className="h-6 w-6 rounded hover:bg-gray-100"
              >
                ◐
              </button>
              <button
                onClick={() => view.current?.home()}
                title={t("mesh.reset")}
                className="h-6 w-6 rounded hover:bg-gray-100"
              >
                ⌂
              </button>
            </div>
          </>
        )}
      </div>
      {scene && scene.times.length > 1 && (
        <div
          className={`flex shrink-0 items-center gap-2 border-t px-2 py-1 text-xs ${
            dark ? "border-gray-700 text-gray-100" : "border-gray-200 text-gray-700"
          }`}
        >
          <button
            onClick={() => setPlaying((p) => !p)}
            className={`w-6 rounded text-sm ${dark ? "hover:bg-gray-700" : "hover:bg-gray-100"}`}
            aria-label={playing ? "pause" : "play"}
            data-testid="scene-play"
          >
            {playing ? "⏸" : "▶"}
          </button>
          <input
            type="range"
            min={0}
            max={scene.times.length - 1}
            value={step}
            onChange={(e) => setStep(Number(e.target.value))}
            className={`min-w-0 flex-1 ${dark ? "accent-blue-400" : "accent-blue-600"}`}
            aria-label={t("scene.time")}
            data-testid="scene-time"
          />
          <span className="w-24 shrink-0 text-right tabular-nums">
            t = {scene.times[step].toPrecision(3)} ({step + 1}/{scene.times.length})
          </span>
          <select
            value={fps}
            onChange={(e) => setFps(Number(e.target.value))}
            className={`rounded border text-xs ${
              dark ? "border-gray-600 bg-gray-800 text-gray-100" : "border-gray-200 bg-white"
            }`}
            aria-label={t("scene.speed")}
          >
            {[2, 6, 12, 24].map((f) => (
              <option key={f} value={f}>
                {f} fps
              </option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
}
