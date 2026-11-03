import { useEffect, useState } from "react";
import {
  backendChoice,
  deviceChoice,
  localAddress,
  localPort,
  probe,
  setDeviceChoice,
  setLocalPort,
  type BackendChoice,
  type Device,
  type Health,
} from "../api";
import { useLang, type Key } from "../i18n";
import GearIcon from "./GearIcon";
import ShellCommand from "./ShellCommand";

type Tab = "local" | "remote" | "subscribed";
type Os = "windows" | "mac" | "linux";

const RELEASES = "https://github.com/jackyko1991/tensorizer/releases/tag/runner-latest";
const RUNNER_ZIP = "https://github.com/jackyko1991/tensorizer/releases/download/runner-latest/tensorizer-runner.zip";
const DOWNLOAD: Record<Os, string> = {
  windows: `irm ${RUNNER_ZIP} -OutFile tensorizer-runner.zip; Expand-Archive tensorizer-runner.zip . -Force; cd tensorizer-runner`,
  mac: `curl -fsSL -o tensorizer-runner.zip ${RUNNER_ZIP} && unzip -o tensorizer-runner.zip && cd tensorizer-runner`,
  linux: `curl -fsSL -o tensorizer-runner.zip ${RUNNER_ZIP} && unzip -o tensorizer-runner.zip && cd tensorizer-runner`,
};
const PIXI_INSTALL: Record<Os, string> = {
  windows: 'powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"',
  mac: "curl -fsSL https://pixi.sh/install.sh | sh",
  linux: "curl -fsSL https://pixi.sh/install.sh | sh",
};

/** Phones and tablets (iPadOS reports itself as a Mac, but with touch): they can't run the local runner. */
export function isMobileDevice(): boolean {
  const ua = navigator.userAgent;
  const hint = (navigator as Navigator & { userAgentData?: { mobile?: boolean } }).userAgentData?.mobile;
  return (
    !!hint || /Android|iPhone|iPad|iPod|Mobile/i.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1)
  );
}

function detectOs(): Os {
  const p = (navigator.userAgent + " " + (navigator.platform ?? "")).toLowerCase();
  return p.includes("win") ? "windows" : p.includes("mac") ? "mac" : "linux";
}

/** The header button: where runs happen now, with a status dot; opens the panel. */
export function BackendButton({ backend, onOpen }: { backend: Health | null | "checking"; onOpen: () => void }) {
  const { t } = useLang();
  const local = backendChoice() === "local";
  const dot = backend === "checking" ? "bg-gray-300" : backend ? "bg-green-500" : "bg-amber-400";
  const device = local && backend && backend !== "checking" ? ` / ${deviceChoice()}` : "";
  return (
    <button
      onClick={onOpen}
      title={t("be.button")}
      data-testid="backend-button"
      className="flex h-8 items-center gap-2 rounded-md border border-blue-400 bg-blue-600 px-2.5 text-xs text-white hover:bg-blue-500"
    >
      <span className={`h-2 w-2 rounded-full ${dot}`} />
      <span className="hidden sm:inline">
        {t(local ? "be.tab.local" : "be.shared")}
        {device}
        {!backend && ` / ${t("be.offline")}`}
      </span>
      <GearIcon className="h-4 w-4" />
    </button>
  );
}

/** ⚙ Backend: where runs happen. Local (this computer) works now; Remote and Subscribed are coming. */
export default function BackendPanel({
  onClose,
  onChoose,
}: {
  onClose: () => void;
  onChoose: (c: BackendChoice) => void;
}) {
  const { t } = useLang();
  const mobile = isMobileDevice();
  const [tab, setTab] = useState<Tab>(backendChoice() === "local" && !mobile ? "local" : "remote");
  useEffect(() => {
    const key = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-gray-900/40 p-0 text-left text-gray-900 sm:p-6"
      onMouseDown={onClose}
    >
      <div
        role="dialog"
        aria-label={t("be.title")}
        data-testid="backend-panel"
        onMouseDown={(e) => e.stopPropagation()}
        className="flex max-h-full w-full flex-col overflow-hidden bg-white shadow-2xl sm:mt-10 sm:max-w-xl sm:rounded-xl"
      >
        <div className="flex items-center justify-between border-b border-gray-200 px-5 py-3">
          <h2 className="text-base font-semibold">{t("be.title")}</h2>
          <button
            onClick={onClose}
            aria-label={t("plot.close")}
            className="text-xl leading-none text-gray-400 hover:text-gray-700"
          >
            ×
          </button>
        </div>
        <div className="flex gap-1 border-b border-gray-200 px-4 text-sm" role="tablist">
          {(["local", "remote", "subscribed"] as const).map((k) => (
            <button
              key={k}
              role="tab"
              aria-selected={tab === k}
              onClick={() => setTab(k)}
              className={`-mb-px border-b-2 px-3 py-2 font-medium ${
                tab === k ? "border-blue-600 text-blue-700" : "border-transparent text-gray-500 hover:text-gray-800"
              }`}
            >
              {t(`be.tab.${k}` as Key)}
              {k === "subscribed" && (
                <span className="ml-1.5 text-[10px] font-normal text-gray-400">{t("be.soon")}</span>
              )}
            </button>
          ))}
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4 text-sm">
          {tab === "local" &&
            (mobile ? (
              <p
                className="rounded-md border border-gray-200 bg-gray-50 px-3 py-3 text-gray-600"
                data-testid="local-mobile"
              >
                📱 {t("be.mobile")}
              </p>
            ) : (
              <LocalTab onUse={() => onChoose("local")} />
            ))}
          {tab === "remote" && <RemoteTab onUse={() => onChoose("server")} />}
          {tab === "subscribed" && <p className="text-gray-600">{t("be.sub.soon")}</p>}
        </div>
      </div>
    </div>
  );
}

function LocalTab({ onUse }: { onUse: () => void }) {
  const { t } = useLang();
  const [port, setPort] = useState(String(localPort()));
  const portNumber = Number(port);
  const portOk = Number.isInteger(portNumber) && portNumber > 0 && portNumber < 65536;
  const address = localAddress(portOk ? portNumber : localPort());
  const [health, setHealth] = useState<Health | null | "checking">("checking");
  const [device, setDevice] = useState<Device>(deviceChoice);
  const [os, setOs] = useState<Os>(detectOs);
  const inUse = backendChoice() === "local";

  // Test now, then keep checking every 3 s until the runner answers (so starting it connects by itself).
  useEffect(() => {
    let alive = true;
    let timer: number | undefined;
    const check = async () => {
      const h = await probe(address);
      if (!alive) return;
      setHealth(h);
      if (!h) timer = window.setTimeout(check, 3000);
    };
    setHealth("checking");
    void check();
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [address]);

  const h = health === "checking" ? null : health;
  const has = (d: Device) => d === "auto" || d === "cpu" || !!h?.devices?.includes(d);
  const pick = (d: Device) => {
    setDevice(d);
    setDeviceChoice(d);
  };

  return (
    <div className="space-y-4">
      <p className="text-gray-600">{t("be.local.intro")}</p>
      <p className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
        ⚠ {t("be.local.warn")}
      </p>

      <div className="flex items-start gap-2">
        <span className="w-20 shrink-0 pt-1 text-gray-500">{t("be.port")}</span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs text-gray-400">localhost:</span>
            <input
              value={port}
              onChange={(e) => setPort(e.target.value.replace(/\D/g, "").slice(0, 5))}
              onBlur={() => portOk && setLocalPort(portNumber)}
              inputMode="numeric"
              aria-label={t("be.port")}
              data-testid="local-port"
              className={`w-20 rounded-md border px-2 py-1 font-mono text-xs ${portOk ? "border-gray-300" : "border-red-400"}`}
            />
            <button
              onClick={() => {
                if (!portOk) return;
                setLocalPort(portNumber);
                setHealth("checking");
                void probe(address).then(setHealth);
              }}
              className="rounded-md border border-gray-300 px-2.5 py-1 text-xs hover:bg-gray-50"
            >
              {t("be.test")}
            </button>
          </div>
          <p className="mt-1 text-[11px] text-gray-500">{t("be.portHint")}</p>
        </div>
      </div>

      <div className="flex items-start gap-2">
        <span className="w-20 shrink-0 text-gray-500">{t("be.status")}</span>
        <div className="text-xs" data-testid="local-status">
          {health === "checking" ? (
            <span className="text-gray-500">…</span>
          ) : h ? (
            <>
              <span className="font-medium text-green-700">● {t("be.connected")}</span>
              <span className="text-gray-500">
                {h.runner && ` / runner ${h.runner}`}
                {h.torch && ` / torch ${h.torch}`}
                {h.os && ` / ${h.os}`}
                {h.gpu && ` / ${h.gpu}`}
              </span>
            </>
          ) : (
            <span className="text-amber-700">● {t("be.waiting")}</span>
          )}
          {h?.nvidia && (
            <p className="mt-1.5 rounded-md bg-amber-50 px-2 py-1.5 text-amber-900" data-testid="cuda-hint">
              {t("be.cudaHint").replace("{gpu}", h.nvidia)} <code className="font-mono">pixi run -e cuda api</code>
            </p>
          )}
        </div>
      </div>

      <div className="flex items-start gap-2">
        <span className="w-20 shrink-0 pt-0.5 text-gray-500">{t("be.device")}</span>
        <div className="space-y-1">
          <div className="flex flex-wrap gap-3 text-xs" role="radiogroup" aria-label={t("be.device")}>
            {(["auto", "cpu", "cuda", "mps"] as const).map((d) => (
              <label key={d} className={`flex items-center gap-1 ${has(d) ? "" : "text-gray-300"}`}>
                <input type="radio" name="device" checked={device === d} disabled={!has(d)} onChange={() => pick(d)} />
                {d === "auto" ? t("be.auto") : d.toUpperCase()}
              </label>
            ))}
          </div>
          {(device === "mps" || h?.devices?.includes("mps")) && (
            <p className="text-[11px] text-gray-500">{t("be.mpsNote")}</p>
          )}
        </div>
      </div>

      <details className="rounded-lg border border-gray-200" open={!h}>
        <summary className="cursor-pointer px-3 py-2 font-medium text-gray-700">{t("be.howto")}</summary>
        <div className="space-y-3 border-t border-gray-100 px-3 py-3 text-xs">
          <div className="flex gap-1">
            {(["windows", "mac", "linux"] as const).map((o) => (
              <button
                key={o}
                onClick={() => setOs(o)}
                className={`rounded px-2 py-0.5 ${os === o ? "bg-blue-50 font-medium text-blue-700" : "text-gray-500 hover:text-gray-800"}`}
              >
                {o === "windows" ? "Windows" : o === "mac" ? "macOS" : "Linux"}
              </button>
            ))}
          </div>
          <Step n={1} label={t("be.step1")} command={PIXI_INSTALL[os]} />
          <Step n={2} label={t("be.step2")} command={DOWNLOAD[os]} />
          <p className="-mt-1.5 text-gray-500">
            {t("be.orRelease")}{" "}
            <a href={RELEASES} target="_blank" rel="noreferrer" className="text-blue-700 hover:underline">
              GitHub Releases ↗
            </a>
          </p>
          <Step n={3} label={t("be.step3")} command="pixi run api" />
          {os !== "mac" && (
            <Step n={4} label={t("be.stepGpu")} command="pixi run -e cuda api" badge={t("be.experimental")} />
          )}
          <p className="text-gray-400">{t("be.later")}</p>
        </div>
      </details>

      <div className="flex justify-end">
        {inUse ? (
          <span className="rounded-md bg-green-50 px-3 py-1.5 text-xs font-medium text-green-700">
            ✓ {t("be.inUse")}
          </span>
        ) : (
          <button
            onClick={() => confirm(t("be.confirm")) && onUse()}
            className="rounded-md bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-700"
            data-testid="use-local"
          >
            {t("be.use")}
          </button>
        )}
      </div>
    </div>
  );
}

function Step({ n, label, command, badge }: { n: number; label: string; command: string; badge?: string }) {
  return (
    <div>
      <p className="mb-1 text-gray-600">
        {n}. {label}
        {badge && (
          <span className="ml-2 rounded bg-amber-100 px-1.5 py-0.5 text-[10px] font-medium text-amber-800">
            {badge}
          </span>
        )}
      </p>
      <ShellCommand text={command} size="text-[11px]" />
    </div>
  );
}

function RemoteTab({ onUse }: { onUse: () => void }) {
  const { t } = useLang();
  const inUse = backendChoice() === "server";
  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-gray-200 p-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="font-medium text-gray-800">{t("be.shared")}</p>
            <p className="text-xs text-gray-500">{t("be.sharedHint")}</p>
          </div>
          {inUse ? (
            <span className="shrink-0 rounded-md bg-green-50 px-3 py-1.5 text-xs font-medium text-green-700">
              ✓ {t("be.inUse")}
            </span>
          ) : (
            <button
              onClick={onUse}
              className="shrink-0 rounded-md border border-gray-300 px-3 py-1.5 text-xs hover:bg-gray-50"
              data-testid="use-shared"
            >
              {t("be.useShared")}
            </button>
          )}
        </div>
      </div>
      <p className="text-gray-600">
        <span className="mr-1.5 rounded bg-gray-100 px-1.5 py-0.5 text-[10px] text-gray-500">{t("be.soon")}</span>
        {t("be.remote.soon")}
      </p>
    </div>
  );
}
