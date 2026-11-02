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

/** An uploaded picture as diagram.svg text: an SVG as is, a PNG/JPG embedded in an SVG of its own size. */
export async function pictureToSvg(file: File): Promise<string> {
  if (file.type === "image/svg+xml" || file.name.toLowerCase().endsWith(".svg")) return file.text();
  if (!/^image\/(png|jpeg)$/.test(file.type)) throw new Error("use an SVG, PNG or JPG picture");
  const dataUrl = await new Promise<string>((ok, fail) => {
    const reader = new FileReader();
    reader.onload = () => ok(String(reader.result));
    reader.onerror = () => fail(reader.error);
    reader.readAsDataURL(file);
  });
  const img = new Image();
  img.src = dataUrl;
  await img.decode();
  const w = img.naturalWidth;
  const h = img.naturalHeight;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}"><image href="${dataUrl}" width="${w}" height="${h}"/></svg>`;
}

/** diagram.svg text ⇄ the data: URI the page shows (UTF-8 safe base64). */
export function svgToDataUri(svg: string): string {
  const bytes = new TextEncoder().encode(svg);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return "data:image/svg+xml;base64," + btoa(binary);
}
export function dataUriToSvg(uri: string): string | null {
  const m = /^data:image\/svg\+xml;base64,(.*)$/.exec(uri);
  if (!m) return null;
  return new TextDecoder().decode(Uint8Array.from(atob(m[1]), (c) => c.charCodeAt(0)));
}
