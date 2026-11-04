/** A 3D view of an STL file (three.js, loaded only when a mesh is opened): shaded, on a floor grid (rotation off until asked for),
 *  orbit / zoom with the mouse or touch, and a small toolbar for colour, grid, axes, wireframe, edges, rotation,
 *  background and a view reset (remembered in the browser). */
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import type { Dataset } from "../types";
import { useLang } from "../i18n";

interface Options {
  color: string;
  grid: boolean;
  axes: boolean;
  wireframe: boolean;
  edges: boolean;
  rotate: boolean;
  dark: boolean;
  surface: boolean;
  flat: boolean;
  glossy: boolean;
  opacity: number;
}
const DEFAULTS: Options = {
  color: "#7aa2e3",
  grid: true,
  axes: false,
  wireframe: false,
  edges: false,
  rotate: false,
  dark: false,
  surface: true,
  flat: false,
  glossy: false,
  opacity: 1,
};
const STORE = "tensorizer.meshView.v2"; // v2: auto-rotate now off by default

function loadOptions(): Options {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(STORE) ?? "{}") };
  } catch {
    return DEFAULTS;
  }
}

/** Scene objects the toolbar changes after the mesh has loaded. */
interface Scene {
  material: THREE.MeshStandardMaterial;
  grid: THREE.GridHelper;
  axes: THREE.AxesHelper;
  edges: THREE.LineSegments;
  controls: OrbitControls;
  renderer: THREE.WebGLRenderer;
  home: () => void;
}

/** Colour stops of the common colormaps, for the legend bar (vertex colours come baked into the PLY). */
const CMAPS: Record<string, string[]> = {
  viridis: ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
  magma: ["#000004", "#51127c", "#b73779", "#fc8961", "#fcfdbf"],
  inferno: ["#000004", "#56106e", "#bb3754", "#f98e09", "#fcffa4"],
  plasma: ["#0d0887", "#7e03a8", "#cc4778", "#f89540", "#f0f921"],
  cividis: ["#00224e", "#35456c", "#666970", "#a59c74", "#fee838"],
  coolwarm: ["#3b4cc0", "#8db0fe", "#dddddd", "#f49a7b", "#b40426"],
  rdbu: ["#67001f", "#d6604d", "#f7f7f7", "#4393c3", "#053061"],
};

export default function MeshView({
  url,
  tall = false,
  fill = false,
  legend,
}: {
  url: string;
  tall?: boolean;
  /** Take the full height of the parent (a flex panel) instead of a fixed height. */
  fill?: boolean;
  legend?: Dataset["legend"];
}) {
  const { t } = useLang();
  const host = useRef<HTMLDivElement>(null);
  const [info, setInfo] = useState<string>("…");
  const [opts, setOpts] = useState<Options>(loadOptions);
  const scene = useRef<Scene | null>(null);
  const optsRef = useRef(opts);
  // ⤢ fills the page with the view (an overlay, so it also works inside frames); Esc or ⤡ brings it back
  const [expanded, setExpanded] = useState(false);
  useEffect(() => {
    if (!expanded) return;
    const key = (e: KeyboardEvent) => e.key === "Escape" && setExpanded(false);
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [expanded]);
  optsRef.current = opts;

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    let stop = false;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);
    const world = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 1000);
    world.add(new THREE.HemisphereLight(0xffffff, 0x8a8f99, 1.1));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(2, 3, 2);
    world.add(sun);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotateSpeed = 1.2;

    const resize = () => {
      const w = el.clientWidth;
      const h = el.clientHeight;
      renderer.setSize(w, h, false);
      renderer.domElement.style.width = "100%";
      renderer.domElement.style.height = "100%";
      camera.aspect = w / Math.max(h, 1);
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(el);
    resize();

    fetch(url)
      .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(String(r.status)))))
      .then((buf) => {
        if (stop) return;
        const ply = /\.ply(\?|$)/i.test(url);
        const geometry = ply ? new PLYLoader().parse(buf) : new STLLoader().parse(buf);
        const coloured = !!geometry.getAttribute("color"); // result meshes carry per-vertex colours
        geometry.computeVertexNormals();
        geometry.computeBoundingBox();
        const box = geometry.boundingBox!;
        const size = box.getSize(new THREE.Vector3());
        const centre = box.getCenter(new THREE.Vector3());
        geometry.translate(-centre.x, -centre.y, -box.min.z); // centred, resting on the floor (z up)
        const material = new THREE.MeshStandardMaterial({
          metalness: 0.1,
          roughness: 0.55,
          side: THREE.DoubleSide,
          vertexColors: coloured,
        });
        material.userData.coloured = coloured;
        const mesh = new THREE.Mesh(geometry, material);
        mesh.rotation.x = -Math.PI / 2; // STL z-up → three.js y-up
        world.add(mesh);
        const edges = new THREE.LineSegments(
          new THREE.EdgesGeometry(geometry, 25),
          new THREE.LineBasicMaterial({ color: 0x1f2937 }),
        );
        edges.rotation.x = -Math.PI / 2;
        world.add(edges);
        const span = Math.max(size.x, size.y, size.z);
        const grid = new THREE.GridHelper(span * 2, 20, 0x9ca3af, 0xd1d5db);
        world.add(grid);
        const axes = new THREE.AxesHelper(span * 0.6); // red x (flow direction), green z-up, blue y
        world.add(axes);
        camera.near = span / 100; // clip planes scaled to the model (STLs come in mm, m, lattice units …)
        camera.far = span * 50;
        camera.updateProjectionMatrix();
        const home = () => {
          const back = 1 / Math.min(1, camera.aspect); // step back in tall, narrow panels so the model fits across
          camera.position.set(-span * 1.4 * back, span * 0.9 * back, span * 1.4 * back); // the front (−x) towards the lower left
          controls.target.set(0, size.z / 2, 0);
          controls.update();
        };
        home();
        scene.current = { material, grid, axes, edges, controls, renderer, home };
        apply(scene.current, optsRef.current);
        const triangles = (geometry.index?.count ?? geometry.attributes.position.count) / 3; // PLY is indexed, STL not
        setInfo(
          `${triangles.toLocaleString()} triangles · ${size.x.toPrecision(3)} × ${size.y.toPrecision(3)} × ${size.z.toPrecision(3)}`,
        );
      })
      .catch(() => !stop && setInfo("could not load the mesh"));

    const tick = () => {
      if (stop) return;
      controls.update();
      renderer.render(world, camera);
      requestAnimationFrame(tick);
    };
    tick();

    return () => {
      stop = true;
      scene.current = null;
      observer.disconnect();
      controls.dispose();
      world.traverse((o) => {
        if (o instanceof THREE.Mesh || o instanceof THREE.LineSegments) {
          o.geometry.dispose();
          (o.material as THREE.Material).dispose();
        }
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [url]);

  // toolbar changes: update the live scene and remember them
  useEffect(() => {
    if (scene.current) apply(scene.current, opts);
    try {
      localStorage.setItem(STORE, JSON.stringify(opts));
    } catch {
      /* storage unavailable */
    }
  }, [opts]);

  type Flag = keyof Omit<Options, "color" | "opacity">;
  const toggle = (key: Flag) => setOpts((o) => ({ ...o, [key]: !o[key] }));
  const tools: { key: Flag; icon: string; label: string }[] = [
    { key: "surface", icon: "▣", label: t("mesh.surface") },
    { key: "flat", icon: "◆", label: t("mesh.flat") },
    { key: "glossy", icon: "✦", label: t("mesh.glossy") },
    { key: "grid", icon: "⊞", label: t("mesh.grid") },
    { key: "axes", icon: "✛", label: t("mesh.axes") },
    { key: "wireframe", icon: "◇", label: t("mesh.wireframe") },
    { key: "edges", icon: "⧈", label: t("mesh.edges") },
    { key: "rotate", icon: "⟳", label: t("mesh.rotate") },
    { key: "dark", icon: "◐", label: t("mesh.dark") },
  ];

  return (
    <div
      className={`rounded ${opts.dark ? "bg-gray-900" : expanded ? "bg-white" : ""} ${
        expanded ? "fixed inset-3 z-50 border border-gray-200 shadow-2xl" : fill ? "relative h-full" : "relative"
      }`}
      data-testid="mesh-frame"
    >
      <div
        ref={host}
        className={`w-full ${expanded || fill ? "h-full" : tall ? "h-[60vh]" : "h-72"}`}
        data-testid="mesh-view"
      />
      <div
        className="absolute top-1 right-1 left-1 ml-auto flex w-fit max-w-full flex-wrap items-center justify-end gap-0.5 rounded-md border border-gray-200 bg-white/90 p-0.5 text-xs shadow-sm"
        data-testid="mesh-tools"
      >
        <label
          title={t("mesh.color")}
          className="flex h-6 w-6 cursor-pointer items-center justify-center rounded hover:bg-gray-100"
        >
          <input
            type="color"
            value={opts.color}
            onChange={(e) => setOpts((o) => ({ ...o, color: e.target.value }))}
            className="h-4 w-4 cursor-pointer border-0 bg-transparent p-0"
            aria-label={t("mesh.color")}
          />
        </label>
        {tools.map((tool) => (
          <button
            key={tool.key}
            onClick={() => toggle(tool.key)}
            title={tool.label}
            aria-label={tool.label}
            aria-pressed={opts[tool.key]}
            data-testid={`mesh-${tool.key}`}
            className={`h-6 w-6 rounded ${opts[tool.key] ? "bg-blue-50 text-blue-700" : "text-gray-500 hover:bg-gray-100"}`}
          >
            {tool.icon}
          </button>
        ))}
        <input
          type="range"
          min={0.15}
          max={1}
          step={0.05}
          value={opts.opacity}
          onChange={(e) => setOpts((o) => ({ ...o, opacity: Number(e.target.value) }))}
          title={`${t("mesh.opacity")}: ${Math.round(opts.opacity * 100)}%`}
          aria-label={t("mesh.opacity")}
          data-testid="mesh-opacity"
          className="mx-1 h-1 w-14 cursor-pointer accent-blue-600"
        />
        <button
          onClick={() => scene.current?.home()}
          title={t("mesh.reset")}
          aria-label={t("mesh.reset")}
          className="h-6 w-6 rounded text-gray-500 hover:bg-gray-100"
        >
          ⌂
        </button>
        <button
          onClick={() => setExpanded((e) => !e)}
          title={t(expanded ? "mesh.shrink" : "mesh.expand")}
          aria-label={t(expanded ? "mesh.shrink" : "mesh.expand")}
          aria-pressed={expanded}
          data-testid="mesh-expand"
          className="h-6 w-6 rounded text-gray-500 hover:bg-gray-100"
        >
          {expanded ? "⤡" : "⤢"}
        </button>
      </div>
      {legend && (
        <div
          className="absolute bottom-6 left-2 rounded border border-gray-200 bg-white/90 px-2 py-1 text-[10px] text-gray-700"
          data-testid="mesh-legend"
        >
          <p className="font-medium">
            {legend.label}
            {legend.unit ? ` (${legend.unit})` : ""}
          </p>
          <div
            className="my-0.5 h-2 w-32 rounded-sm"
            style={{
              background: `linear-gradient(to right, ${(CMAPS[(legend.cmap ?? "viridis").toLowerCase()] ?? CMAPS.viridis).join(",")})`,
            }}
          />
          <p className="flex justify-between tabular-nums">
            <span>{legend.min.toPrecision(3)}</span>
            <span>{legend.max.toPrecision(3)}</span>
          </p>
        </div>
      )}
      {info === "…" && (
        <div className="absolute inset-0 flex items-center justify-center" data-testid="mesh-loading">
          <span className="h-8 w-8 animate-spin rounded-full border-2 border-gray-200 border-t-blue-600" />
        </div>
      )}
      <p className={`absolute bottom-1 left-2 text-[10px] ${opts.dark ? "text-gray-400" : "text-gray-500"}`}>
        {info === "…" ? "" : info}
      </p>
    </div>
  );
}

function apply(s: Scene, o: Options) {
  s.material.color.set(s.material.userData.coloured ? "#ffffff" : o.color); // keep baked colours true
  s.material.wireframe = o.wireframe;
  s.material.visible = o.surface || o.wireframe;
  s.material.flatShading = o.flat;
  s.material.metalness = o.glossy ? 0.6 : 0.1;
  s.material.roughness = o.glossy ? 0.2 : 0.55;
  s.material.transparent = o.opacity < 1;
  s.material.opacity = o.opacity;
  s.material.depthWrite = o.opacity >= 1;
  s.material.needsUpdate = true; // flat shading and transparency need a recompiled material
  s.grid.visible = o.grid;
  s.axes.visible = o.axes;
  s.edges.visible = o.edges;
  s.controls.autoRotate = o.rotate;
  s.renderer.setClearColor(o.dark ? 0x111827 : 0x000000, o.dark ? 1 : 0);
}
