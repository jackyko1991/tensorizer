/** A 3D view of an STL file (three.js, loaded only when a mesh is opened): shaded, on a floor grid, auto-rotating,
 *  orbit / zoom with the mouse or touch. */
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";

export default function MeshView({ url, tall = false }: { url: string; tall?: boolean }) {
  const host = useRef<HTMLDivElement>(null);
  const [info, setInfo] = useState<string>("…");

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    let stop = false;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 1000);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f99, 1.1));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(2, 3, 2);
    scene.add(sun);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 1.2;
    renderer.domElement.addEventListener("pointerdown", () => (controls.autoRotate = false), { once: true });

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
        const mesh = new THREE.Mesh(
          geometry,
          new THREE.MeshStandardMaterial({ color: 0x7aa2e3, metalness: 0.1, roughness: 0.55, side: THREE.DoubleSide }),
        );
        mesh.rotation.x = -Math.PI / 2; // STL z-up → three.js y-up
        scene.add(mesh);
        const span = Math.max(size.x, size.y, size.z);
        const grid = new THREE.GridHelper(span * 2, 20, 0x9ca3af, 0xd1d5db);
        scene.add(grid);
        camera.near = span / 100; // clip planes scaled to the model (STLs come in mm, m, lattice units …)
        camera.far = span * 50;
        camera.updateProjectionMatrix();
        camera.position.set(-span * 1.4, span * 0.9, span * 1.4); // the front (−x) towards the lower left
        controls.target.set(0, size.z / 2, 0);
        controls.update();
        const triangles = geometry.attributes.position.count / 3;
        setInfo(
          `${triangles.toLocaleString()} triangles · ${size.x.toPrecision(3)} × ${size.y.toPrecision(3)} × ${size.z.toPrecision(3)}`,
        );
      })
      .catch(() => !stop && setInfo("could not load the mesh"));

    const tick = () => {
      if (stop) return;
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(tick);
    };
    tick();

    return () => {
      stop = true;
      observer.disconnect();
      controls.dispose();
      scene.traverse((o) => {
        if (o instanceof THREE.Mesh) {
          o.geometry.dispose();
          (o.material as THREE.Material).dispose();
        }
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [url]);

  return (
    <div className="relative">
      <div ref={host} className={`w-full ${tall ? "h-[60vh]" : "h-72"}`} data-testid="mesh-view" />
      {info === "…" && (
        <div className="absolute inset-0 flex items-center justify-center" data-testid="mesh-loading">
          <span className="h-8 w-8 animate-spin rounded-full border-2 border-gray-200 border-t-blue-600" />
        </div>
      )}
      <p className="absolute bottom-1 left-2 text-[10px] text-gray-500">{info === "…" ? "" : info}</p>
    </div>
  );
}
