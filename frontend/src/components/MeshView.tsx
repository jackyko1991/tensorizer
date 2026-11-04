/** A 3D view of an STL file (three.js, loaded only when a mesh is opened): shaded, on a floor grid (rotation off until asked for),
 *  orbit / zoom with the mouse or touch, and a small toolbar for colour, grid, axes, wireframe, edges, rotation,
 *  background and a view reset (remembered in the browser). */
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
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

export default function MeshView({ url, tall = false }: { url: string; tall?: boolean }) {
  const { t } = useLang();
  const host = useRef<HTMLDivElement>(null);
  const [info, setInfo] = useState<string>("…");
  const [opts, setOpts] = useState<Options>(loadOptions);
  const scene = useRef<Scene | null>(null);
  const optsRef = useRef(opts);
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
        const geometry = new STLLoader().parse(buf);
        geometry.computeVertexNormals();
        geometry.computeBoundingBox();
        const box = geometry.boundingBox!;
        const size = box.getSize(new THREE.Vector3());
        const centre = box.getCenter(new THREE.Vector3());
        geometry.translate(-centre.x, -centre.y, -box.min.z); // centred, resting on the floor (z up)
        const material = new THREE.MeshStandardMaterial({ metalness: 0.1, roughness: 0.55, side: THREE.DoubleSide });
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
          camera.position.set(-span * 1.4, span * 0.9, span * 1.4); // the front (−x) towards the lower left
          controls.target.set(0, size.z / 2, 0);
          controls.update();
        };
        home();
        scene.current = { material, grid, axes, edges, controls, renderer, home };
        apply(scene.current, optsRef.current);
        const triangles = geometry.attributes.position.count / 3;
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
    <div className={`relative rounded ${opts.dark ? "bg-gray-900" : ""}`}>
      <div ref={host} className={`w-full ${tall ? "h-[60vh]" : "h-72"}`} data-testid="mesh-view" />
      <div
        className="absolute top-1 right-1 flex items-center gap-0.5 rounded-md border border-gray-200 bg-white/90 p-0.5 text-xs shadow-sm"
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
      </div>
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
  s.material.color.set(o.color);
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
