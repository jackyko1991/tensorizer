/** Download a problem's physical picture (diagram.svg) as SVG, or rasterised to PNG / JPG in the browser. */

export type DiagramFormat = "svg" | "png" | "jpg";
export const DIAGRAM_FORMATS: DiagramFormat[] = ["svg", "png", "jpg"];

function save(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** The SVG's drawing size: its width/height, else its viewBox (the library uses viewBox 0 0 240 160). */
function svgSize(text: string): { w: number; h: number } {
  const doc = new DOMParser().parseFromString(text, "image/svg+xml").documentElement;
  const view = doc
    .getAttribute("viewBox")
    ?.split(/[\s,]+/)
    .map(Number);
  const w = parseFloat(doc.getAttribute("width") ?? "") || view?.[2] || 240;
  const h = parseFloat(doc.getAttribute("height") ?? "") || view?.[3] || 160;
  return { w, h };
}

/** src: the diagram's URL (a generated asset or a Workspace data: URI); name: the file name without extension. */
export async function exportDiagram(src: string, name: string, format: DiagramFormat, scale = 4): Promise<void> {
  const text = await (await fetch(src)).text();
  const svg = new Blob([text], { type: "image/svg+xml" });
  if (format === "svg") return save(svg, `${name}.svg`);

  const { w, h } = svgSize(text);
  const url = URL.createObjectURL(svg);
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    const canvas = Object.assign(document.createElement("canvas"), { width: w * scale, height: h * scale });
    const ctx = canvas.getContext("2d")!;
    if (format === "jpg") {
      ctx.fillStyle = "#ffffff"; // JPG has no transparency
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    const type = format === "png" ? "image/png" : "image/jpeg";
    const blob = await new Promise<Blob | null>((ok) => canvas.toBlob(ok, type, 0.92));
    if (blob) save(blob, `${name}.${format}`);
  } finally {
    URL.revokeObjectURL(url);
  }
}
