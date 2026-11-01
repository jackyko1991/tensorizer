import { useRef, useState } from "react";
import { useLang } from "../i18n";

export const MAX_SCREENSHOT_BYTES = 3 * 1024 * 1024;
const MAX_WIDTH = 1600;

export class CaptureError extends Error {
  constructor(public kind: "unsupported" | "denied") {
    super(kind);
  }
}

/** The PNG's size in bytes, from its data: URL. */
export const pngBytes = (dataUrl: string) =>
  Math.floor(((dataUrl.length - dataUrl.indexOf(",") - 1) * 3) / 4);

/** One frame of the current tab (the user picks it in the browser's share dialog), at most 1600 px wide.
 *  Every track is stopped right after the frame is taken. */
export async function captureTab(): Promise<string> {
  if (!navigator.mediaDevices?.getDisplayMedia)
    throw new CaptureError("unsupported");
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getDisplayMedia({
      video: { displaySurface: "browser" },
      audio: false,
      preferCurrentTab: true,
    } as DisplayMediaStreamOptions);
  } catch {
    throw new CaptureError("denied");
  }
  try {
    const video = document.createElement("video");
    video.srcObject = stream;
    video.muted = true;
    await video.play();
    if (!video.videoWidth)
      await new Promise((r) =>
        video.addEventListener("resize", r, { once: true }),
      );
    const scale = Math.min(1, MAX_WIDTH / video.videoWidth);
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    canvas
      .getContext("2d")!
      .drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/png");
  } finally {
    stream.getTracks().forEach((track) => track.stop());
  }
}

/** Cut a rectangle (in natural image pixels) out of a PNG data URL. */
async function crop(
  dataUrl: string,
  r: { x: number; y: number; w: number; h: number },
): Promise<string> {
  const img = new Image();
  img.src = dataUrl;
  await img.decode();
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(r.w));
  canvas.height = Math.max(1, Math.round(r.h));
  canvas
    .getContext("2d")!
    .drawImage(img, r.x, r.y, r.w, r.h, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/png");
}

/** The enlarged screenshot: drag on it to crop, "Undo crop" to go back to the original. */
export function ScreenshotPreview({
  image,
  original,
  onChange,
  onClose,
}: {
  image: string;
  original: string;
  onChange: (dataUrl: string) => void;
  onClose: () => void;
}) {
  const { t } = useLang();
  const imgRef = useRef<HTMLImageElement>(null);
  const [drag, setDrag] = useState<{
    x0: number;
    y0: number;
    x1: number;
    y1: number;
  } | null>(null);

  const point = (e: React.PointerEvent) => {
    const box = imgRef.current!.getBoundingClientRect();
    return {
      x: Math.min(Math.max(e.clientX - box.left, 0), box.width),
      y: Math.min(Math.max(e.clientY - box.top, 0), box.height),
    };
  };

  async function finish() {
    const img = imgRef.current;
    if (!drag || !img) return setDrag(null);
    const [x, y] = [Math.min(drag.x0, drag.x1), Math.min(drag.y0, drag.y1)];
    const [w, h] = [Math.abs(drag.x1 - drag.x0), Math.abs(drag.y1 - drag.y0)];
    setDrag(null);
    if (w < 8 || h < 8) return;
    const k = img.naturalWidth / img.getBoundingClientRect().width;
    onChange(await crop(image, { x: x * k, y: y * k, w: w * k, h: h * k }));
  }

  return (
    <div
      role="dialog"
      aria-label={t("ag.screenshot")}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      onClick={onClose}
    >
      <div
        className="max-h-full max-w-5xl space-y-2 rounded-lg bg-white p-3 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2 text-xs text-gray-600">
          <span className="font-medium text-gray-800">
            {t("ag.screenshot")}
          </span>
          <span>{t("ag.cropHint")}</span>
          {image !== original && (
            <button
              onClick={() => onChange(original)}
              className="rounded border border-gray-300 px-2 py-0.5 hover:bg-gray-100"
            >
              {t("ag.cropReset")}
            </button>
          )}
          <button
            onClick={onClose}
            className="ml-auto rounded px-2 py-0.5 hover:bg-gray-100"
            aria-label={t("ag.close")}
          >
            ✕
          </button>
        </div>
        <div
          className="relative cursor-crosshair touch-none select-none"
          onPointerDown={(e) => {
            const p = point(e);
            (e.target as HTMLElement).setPointerCapture(e.pointerId);
            setDrag({ x0: p.x, y0: p.y, x1: p.x, y1: p.y });
          }}
          onPointerMove={(e) =>
            drag && setDrag({ ...drag, x1: point(e).x, y1: point(e).y })
          }
          onPointerUp={finish}
        >
          <img
            ref={imgRef}
            src={image}
            alt={t("ag.screenshot")}
            draggable={false}
            className="max-h-[75vh] max-w-full rounded border border-gray-200"
          />
          {drag && (
            <div
              className="pointer-events-none absolute border-2 border-blue-500 bg-blue-500/10"
              style={{
                left: Math.min(drag.x0, drag.x1),
                top: Math.min(drag.y0, drag.y1),
                width: Math.abs(drag.x1 - drag.x0),
                height: Math.abs(drag.y1 - drag.y0),
              }}
            />
          )}
        </div>
      </div>
    </div>
  );
}
