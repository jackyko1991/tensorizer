import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { spawn } from "node:child_process";
import { createReadStream, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

const ROOT = path.resolve(fileURLToPath(new URL(".", import.meta.url)), "..");
const PROBLEMS = path.join(ROOT, "problems");
const GENERATED = path.join(ROOT, "frontend", "public", "generated");
const TYPES: Record<string, string> = {
  ".json": "application/json",
  ".gif": "image/gif",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".txt": "text/plain; charset=utf-8",
};

/** Run scripts/build_assets.py with the given flags; resolves to true on success. */
function buildAssets(args: string[]): Promise<boolean> {
  return new Promise((resolve) => {
    const proc = spawn("python", ["scripts/build_assets.py", ...args], { cwd: ROOT, stdio: "inherit" });
    proc.on("close", (code) => resolve(code === 0));
    proc.on("error", () => resolve(false));
  });
}

/**
 * Dev-only live updates for problem packages. On any edit under problems/ the manifest is refreshed
 * (≈1 s) and the page is told to re-fetch it in place — no page reload, so a running job, the terminal
 * and edited code are kept. The edited problem's GIF is then re-rendered in the background.
 *
 * public/generated is served here straight from disk and kept out of Vite's watcher: otherwise every
 * GIF written there would make Vite reload the page.
 */
function problemsLiveUpdate(): Plugin {
  let timer: NodeJS.Timeout | undefined;
  const pending = new Set<string>();
  let rendering = Promise.resolve();

  return {
    name: "tensorizer-problems-live-update",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use("/generated", (req, res, next) => {
        const file = path.join(GENERATED, decodeURIComponent((req.url ?? "/").split("?")[0]));
        if (!file.startsWith(GENERATED + path.sep)) return next();
        try {
          if (!statSync(file).isFile()) return next();
        } catch {
          return next();
        }
        res.setHeader("Content-Type", TYPES[path.extname(file)] ?? "application/octet-stream");
        res.setHeader("Cache-Control", "no-cache");
        createReadStream(file).pipe(res);
      });

      server.watcher.add(PROBLEMS);
      const notify = () => server.ws.send({ type: "custom", event: "tensorizer:manifest" });

      const onChange = (file: string) => {
        const rel = path.relative(PROBLEMS, file).split(path.sep);
        if (rel[0].startsWith("..") || !/\.(py|yaml|svg)$/.test(file)) return;
        // Only code or meta.yaml directly in problems/<category>/<slug>/ re-renders the GIF. Edits in _common
        // refresh code views without re-rendering everything; diagram.svg and i18n/*.yaml only need the manifest.
        const rendersOutput = rel.length === 3 && !rel[0].startsWith("_") && /(\.py|meta\.yaml)$/.test(file);
        if (rendersOutput) pending.add(`${rel[0]}/${rel[1]}`);
        clearTimeout(timer);
        timer = setTimeout(async () => {
          const ids = [...pending];
          pending.clear();
          if (await buildAssets(["--manifest-only"])) notify();
          for (const id of ids) {
            rendering = rendering.then(async () => {
              server.config.logger.info(`[tensorizer] re-rendering ${id}…`, { timestamp: true });
              if (await buildAssets(["--only", id])) notify();
            });
          }
        }, 300);
      };
      server.watcher.on("change", onChange);
      server.watcher.on("add", onChange);
    },
  };
}

export default defineConfig({
  // Relative base so the build works from any path (localhost now, GitHub Pages later).
  base: "./",
  build: { chunkSizeWarningLimit: 1500 },
  plugins: [react(), tailwindcss(), problemsLiveUpdate()],
  server: {
    proxy: { "/api": "http://localhost:8000" },
    watch: { ignored: ["**/public/generated/**"] },
  },
});
