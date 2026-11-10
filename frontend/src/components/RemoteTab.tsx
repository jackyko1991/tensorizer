import { useEffect, useState, type ReactNode } from "react";
import { backendChoice, refreshBackend } from "../api";
import { useLang, type Key } from "../i18n";
import {
  activeServerId,
  addServer,
  getToken,
  loadServers,
  normalizeUrl,
  removeServer,
  setActiveServerId,
  testConnection,
  updateServer,
  type Server,
  type TestResult,
} from "../servers";
import ShellCommand from "./ShellCommand";

const DOCS = "https://github.com/jackyko1991/tensorizer/blob/main/docs/remote-server.md";
const INSTALL = "curl -fsSL https://jackyko1991.github.io/tensorizer/install.sh | sudo sh -s --";
// where each connection problem is explained in the full guide
const PROBLEM_DOC: Record<string, string> = {
  mixed: "#how-it-fits-together",
  unreachable: "#troubleshooting",
  cors: "#troubleshooting",
  "not-runner": "#troubleshooting",
  token: "#security",
  rate: "#security",
  http: "#troubleshooting",
  url: "#connect-the-website",
};
const PROBLEM_KEY: Record<string, Key> = {
  mixed: "be.rs.p.mixed",
  unreachable: "be.rs.p.unreachable",
  cors: "be.rs.p.cors",
  "not-runner": "be.rs.p.notRunner",
  token: "be.rs.p.token",
  rate: "be.rs.p.rate",
  http: "be.rs.p.http",
  url: "be.rs.p.url",
};

type Status = TestResult | "testing" | undefined;

/** ⚙ Backend → Remote: the visitor's own servers (add, edit, remove, test, use) and how to set one up. */
export default function RemoteTab({ onUse }: { onUse: () => void }) {
  const { t } = useLang();
  const [servers, setServers] = useState<Server[]>(loadServers);
  const [active, setActive] = useState<string | null>(() => (backendChoice() === "remote" ? activeServerId() : null));
  const [editing, setEditing] = useState<string | "new" | null>(null);
  const [status, setStatus] = useState<Record<string, Status>>({});

  const test = async (s: Server) => {
    setStatus((m) => ({ ...m, [s.id]: "testing" }));
    const r = await testConnection(s.url, getToken(s.id));
    setStatus((m) => ({ ...m, [s.id]: r }));
  };
  // test every server once when the tab opens (one request each to its own address)
  useEffect(() => {
    loadServers().forEach((s) => void test(s));
  }, []);

  const reload = () => {
    setServers(loadServers());
    refreshBackend();
  };
  const use = (s: Server) => {
    setActiveServerId(s.id);
    setActive(s.id);
    onUse();
  };

  return (
    <div className="space-y-4">
      <p className="text-gray-600">{t("be.rs.intro")}</p>

      <div className="space-y-2" data-testid="remote-servers">
        <div className="flex items-center justify-between gap-2">
          <h3 className="font-medium text-gray-800">{t("be.rs.mine")}</h3>
          {editing !== "new" && (
            <button
              onClick={() => setEditing("new")}
              className="rounded-md bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-700"
              data-testid="add-server"
            >
              + {t("be.rs.add")}
            </button>
          )}
        </div>
        {editing === "new" && (
          <ServerForm
            onCancel={() => setEditing(null)}
            onSave={(v) => {
              const s = addServer(v);
              if (!s) return false;
              setEditing(null);
              reload();
              void test(s);
              return true;
            }}
          />
        )}
        {servers.length === 0 && editing !== "new" && <p className="text-xs text-gray-500">{t("be.rs.none")}</p>}
        {servers.map((s) =>
          editing === s.id ? (
            <ServerForm
              key={s.id}
              server={s}
              onCancel={() => setEditing(null)}
              onSave={(v) => {
                const next = updateServer(s.id, { ...v, token: v.token || undefined });
                if (!next) return false;
                setEditing(null);
                reload();
                void test(next);
                return true;
              }}
            />
          ) : (
            <ServerCard
              key={s.id}
              server={s}
              status={status[s.id]}
              inUse={active === s.id}
              onTest={() => void test(s)}
              onEdit={() => setEditing(s.id)}
              onUse={() => use(s)}
              onRemove={() => {
                if (!confirm(t("be.rs.removeConfirm"))) return;
                const wasActive = active === s.id;
                removeServer(s.id);
                reload();
                if (wasActive) {
                  setActive(null);
                  onUse(); // the page falls back to this device
                }
              }}
            />
          ),
        )}
      </div>

      <SetupGuide />
    </div>
  );
}

function ServerForm({
  server,
  onSave,
  onCancel,
}: {
  server?: Server;
  onSave: (v: { name: string; url: string; token: string; remember: boolean }) => boolean;
  onCancel: () => void;
}) {
  const { t } = useLang();
  const [name, setName] = useState(server?.name ?? "");
  const [url, setUrl] = useState(server?.url ?? "");
  const [token, setToken] = useState("");
  const [remember, setRemember] = useState(server?.remember ?? false);
  const [bad, setBad] = useState(false);
  const save = () => {
    if (!normalizeUrl(url) || !onSave({ name, url, token, remember })) setBad(true);
  };
  const field = "w-full rounded-md border border-gray-300 px-2 py-1 text-xs";
  return (
    <form
      className="space-y-2 rounded-lg border border-blue-200 bg-blue-50/40 p-3 text-xs"
      onSubmit={(e) => {
        e.preventDefault();
        save();
      }}
      data-testid="server-form"
    >
      <label className="block">
        <span className="text-gray-600">{t("be.rs.name")}</span>
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder={t("be.rs.namePh")} className={field} />
      </label>
      <label className="block">
        <span className="text-gray-600">{t("be.rs.url")}</span>
        <input
          value={url}
          onChange={(e) => (setUrl(e.target.value), setBad(false))}
          placeholder="https://runner.example.org"
          inputMode="url"
          autoCapitalize="off"
          spellCheck={false}
          className={`${field} font-mono ${bad ? "border-red-400" : ""}`}
          data-testid="server-url"
        />
        {bad && <span className="text-red-600">{t("be.rs.badUrl")}</span>}
      </label>
      <label className="block">
        <span className="text-gray-600">{t("be.rs.token")}</span>
        <input
          type="password"
          value={token}
          onChange={(e) => setToken(e.target.value)}
          placeholder={server && getToken(server.id) ? t("be.rs.tokenKeep") : ""}
          autoComplete="off"
          spellCheck={false}
          className={`${field} font-mono`}
          data-testid="server-token"
        />
      </label>
      <label className="flex items-start gap-1.5">
        <input type="checkbox" checked={remember} onChange={(e) => setRemember(e.target.checked)} className="mt-0.5" />
        <span>
          {t("be.rs.remember")}
          <span className="block text-[11px] text-gray-500">{t("be.rs.rememberHint")}</span>
        </span>
      </label>
      <div className="flex justify-end gap-2">
        <button type="button" onClick={onCancel} className="rounded-md border border-gray-300 px-3 py-1 hover:bg-white">
          {t("be.rs.cancel")}
        </button>
        <button
          type="submit"
          className="rounded-md bg-blue-600 px-3 py-1 font-medium text-white hover:bg-blue-700"
          data-testid="server-save"
        >
          {t("be.rs.save")}
        </button>
      </div>
    </form>
  );
}

function ServerCard({
  server,
  status,
  inUse,
  onTest,
  onEdit,
  onUse,
  onRemove,
}: {
  server: Server;
  status: Status;
  inUse: boolean;
  onTest: () => void;
  onEdit: () => void;
  onUse: () => void;
  onRemove: () => void;
}) {
  const { t } = useLang();
  const dot =
    status === undefined || status === "testing"
      ? "bg-gray-300"
      : status.ok
        ? "bg-green-500"
        : status.problem === "token" || status.problem === "rate"
          ? "bg-amber-400"
          : "bg-red-500";
  const link = "text-blue-700 hover:underline";
  return (
    <div className="rounded-lg border border-gray-200 p-3" data-testid="server-card">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${dot}`} />
        <div className="min-w-0 flex-1">
          <p className="truncate font-medium text-gray-800">{server.name}</p>
          <p className="truncate font-mono text-[11px] text-gray-500">{server.url}</p>
        </div>
        {inUse ? (
          <span className="shrink-0 rounded-md bg-green-50 px-3 py-1.5 text-xs font-medium text-green-700">
            ✓ {t("be.inUse")}
          </span>
        ) : (
          <button
            onClick={onUse}
            className="shrink-0 rounded-md bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-700"
            data-testid="use-server"
          >
            {t("be.rs.use")}
          </button>
        )}
      </div>
      <div className="mt-2 text-xs" data-testid="server-status">
        {status === "testing" && <span className="text-gray-500">{t("be.rs.testing")}</span>}
        {status && status !== "testing" && status.ok && (
          <p>
            <span className="font-medium text-green-700">● {t("be.connected")}</span>
            <span className="text-gray-600">
              {status.health.runner && ` / runner ${status.health.runner}`}
              {status.health.devices && ` / ${status.health.devices.join(", ")}`}
              {status.health.gpu && ` (${status.health.gpu})`}
              {status.queue &&
                ` / ${t("be.rs.queue")
                  .replace("{running}", String(status.queue.running))
                  .replace("{waiting}", String(status.queue.waiting))}`}
            </span>
            <span className="mt-0.5 block text-gray-500">
              {status.health.shared ? t("be.rs.shared") : t("be.rs.private")} ·{" "}
              {status.health.code_exec ? t("be.rs.codeOn") : t("be.rs.codeOff")}
              {status.health.auth === "none" && <span className="text-amber-700"> · ⚠ {t("be.rs.open")}</span>}
            </span>
          </p>
        )}
        {status && status !== "testing" && !status.ok && (
          <p className="text-red-700">
            {t(PROBLEM_KEY[status.problem]).replace("{status}", String(status.status ?? ""))}{" "}
            <a href={DOCS + PROBLEM_DOC[status.problem]} target="_blank" rel="noreferrer" className={link}>
              {t("be.rs.help")} ↗
            </a>
          </p>
        )}
      </div>
      <div className="mt-2 flex flex-wrap gap-3 text-xs">
        <button onClick={onTest} className={link} data-testid="test-server">
          {t("be.rs.testConn")}
        </button>
        <button onClick={onEdit} className={link}>
          {t("be.rs.edit")}
        </button>
        <button onClick={onRemove} className="text-red-600 hover:underline">
          {t("be.rs.remove")}
        </button>
      </div>
    </div>
  );
}

function Step({ text, commands }: { text: string; commands?: string[] }) {
  return (
    <li className="space-y-1">
      <p className="text-gray-700">{text}</p>
      {commands?.map((c) => <ShellCommand key={c} text={c} size="text-[11px]" />)}
    </li>
  );
}

function Path({ title, children }: { title: string; children: ReactNode }) {
  return (
    <details className="rounded-md border border-gray-200 bg-white">
      <summary className="cursor-pointer px-3 py-2 font-medium text-gray-800">{title}</summary>
      <div className="space-y-3 border-t border-gray-100 px-3 py-3">{children}</div>
    </details>
  );
}

const TUNNEL = "ssh -N -L 8765:localhost:8765 you@machine";
const TUNNEL_JUMP = "ssh -N -J you@jumphost -L 8765:localhost:8765 you@machine";
const PIXI_INSTALL = "curl -fsSL https://pixi.sh/install.sh | sh";
const CLONE = "git clone https://github.com/jackyko1991/tensorizer && cd tensorizer && pixi install";
const MAKE_TOKEN = 'export TENSORIZER_TOKEN=$(openssl rand -hex 32); echo "$TENSORIZER_TOKEN"';
const SERVE_LOCAL = "pixi run python -m tensorizer_physics.serve --host 127.0.0.1 --port 8765";

/** Make a runner on 127.0.0.1:8765 reachable over HTTPS: Tailscale, or a Cloudflare Tunnel (paths 2 and 5). */
function ReachSteps() {
  const { t } = useLang();
  return (
    <>
      <Step text={t("be.g2.ts1")} commands={["curl -fsSL https://tailscale.com/install.sh | sh", "sudo tailscale up"]} />
      <Step text={t("be.g2.ts2")} />
      <Step text={t("be.g2.ts3")} commands={["sudo tailscale serve --bg 8765", "tailscale serve status"]} />
      <Step text={t("be.g2.ts4")} />
      <p className="-mt-1 text-gray-500">{t("be.g2.s2b")}</p>
      <Step text={t("be.g2.s3")} commands={["cloudflared tunnel --url http://127.0.0.1:8765"]} />
      <Step text={t("be.g2.s4")} />
    </>
  );
}

/** Five set-up paths for non-experts, each step with a copyable command; the full guide is on GitHub. */
function SetupGuide() {
  const { t } = useLang();
  return (
    <div className="space-y-2 rounded-lg border border-gray-200 bg-gray-50 p-3 text-xs" data-testid="setup-guide">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-medium text-gray-800">{t("be.rs.guide")}</h3>
        <a href={DOCS} target="_blank" rel="noreferrer" className="text-blue-700 hover:underline">
          {t("be.rs.fullGuide")} ↗
        </a>
      </div>
      <Path title={t("be.g1.title")}>
        <ol className="list-decimal space-y-2 pl-4">
          <Step text={t("be.g1.s1")} />
          <Step text={t("be.g1.s2")} />
          <Step text={t("be.g1.s3")} />
          <Step text={t("be.g1.s4")} commands={["ssh root@<server-ip>"]} />
          <Step text={t("be.g1.s5")} commands={[`${INSTALL} --sslip`]} />
          <Step text={t("be.g1.s5b")} commands={[`${INSTALL} --domain runner.example.org`]} />
          <Step text={t("be.g1.s5c")} commands={[`${INSTALL} --sslip --gpu`, "nvidia-smi"]} />
          <Step text={t("be.g1.s6")} />
        </ol>
        <p className="text-gray-500">{t("be.g.armNote")}</p>
      </Path>
      <Path title={t("be.g2.title")}>
        <ol className="list-decimal space-y-2 pl-4">
          <Step text={t("be.g2.s1")} commands={[`${INSTALL} --no-tls`]} />
          <ReachSteps />
        </ol>
        <p className="text-gray-500">{t("be.g.armNote")}</p>
      </Path>
      <Path title={t("be.g3.title")}>
        <p className="rounded-md border border-amber-300 bg-amber-50 px-2 py-1.5 font-medium text-amber-900" data-testid="vpn-policy">
          ⚠ {t("be.g3.policy")}
        </p>
        <p className="text-gray-600">{t("be.g3.intro")}</p>
        <p className="font-medium text-gray-800">{t("be.g3.a")}</p>
        <ol className="list-decimal space-y-2 pl-4">
          <Step text={t("be.g3.a1")} commands={[`${INSTALL} --no-tls`]} />
          <Step text={t("be.g3.a2")} commands={[TUNNEL, TUNNEL_JUMP]} />
          <Step text={t("be.g3.a3")} />
          <Step text={t("be.g3.a4")} />
        </ol>
        <p className="font-medium text-gray-800">{t("be.g3.b")}</p>
        <ol className="list-decimal space-y-2 pl-4">
          <Step text={t("be.g3.b1")} commands={[PIXI_INSTALL, CLONE]} />
          <Step text={t("be.g3.b2")} commands={[MAKE_TOKEN, SERVE_LOCAL]} />
          <Step text={t("be.g3.b3")} />
        </ol>
        <p className="rounded-md bg-amber-50 px-2 py-1.5 text-amber-900">⚠ {t("be.g3.b4")}</p>
      </Path>
      <Path title={t("be.g4.title")}>
        <p className="text-gray-600">{t("be.g4.why")}</p>
        <ol className="list-decimal space-y-2 pl-4">
          <Step text={t("be.g4.s1")} commands={[PIXI_INSTALL, CLONE]} />
        </ol>
        <p className="font-medium text-gray-800">{t("be.g4.a")}</p>
        <ol className="list-decimal space-y-2 pl-4">
          <Step text={t("be.g4.a1")} commands={["pixi run python -m tensorizer_physics.serve --port 8765"]} />
          <Step text={t("be.g4.a2")} />
        </ol>
        <p className="text-gray-500">{t("be.g4.f64")}</p>
        <p className="font-medium text-gray-800">{t("be.g4.b")}</p>
        <ol className="list-decimal space-y-2 pl-4">
          <Step text={t("be.g4.b1")} commands={[MAKE_TOKEN, SERVE_LOCAL]} />
          <Step text={t("be.g4.b2")} commands={["tailscale serve --bg 8765", "tailscale serve status"]} />
          <Step text={t("be.g4.b3")} />
        </ol>
      </Path>
      <Path title={t("be.g5.title")}>
        <p className="text-gray-600">{t("be.g5.req")}</p>
        <ol className="list-decimal space-y-2 pl-4">
          <Step text={t("be.g5.s1")} commands={[`${INSTALL} --no-tls`]} />
        </ol>
        <p className="font-medium text-gray-800">{t("be.g5.s2")}</p>
        <ol className="list-decimal space-y-2 pl-4">
          <ReachSteps />
        </ol>
        <p className="font-medium text-gray-800">{t("be.g5.b")}</p>
        <ol className="list-decimal space-y-2 pl-4">
          <Step text={t("be.g3.b1")} commands={[PIXI_INSTALL, CLONE]} />
          <Step text={t("be.g3.b2")} commands={[MAKE_TOKEN, SERVE_LOCAL]} />
          <Step text={t("be.g5.b3")} />
        </ol>
      </Path>
      <div className="pt-1">
        <p className="font-medium text-gray-800">{t("be.rs.secTitle")}</p>
        <ul className="list-disc space-y-0.5 pl-4 text-gray-600">
          <li>{t("be.rs.sec1")}</li>
          <li>{t("be.rs.sec2")}</li>
          <li>{t("be.rs.sec3")}</li>
        </ul>
      </div>
    </div>
  );
}
