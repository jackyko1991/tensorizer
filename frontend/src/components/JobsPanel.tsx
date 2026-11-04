import { useEffect, useRef, useState } from "react";
import {
  cancelJob,
  jobActive,
  JOBS_CHANGED,
  listJobs,
  requestOpenJob,
  type JobInfo,
  type JobStatus,
} from "../api";
import { useLang, type Key } from "../i18n";

const NOTIFY_KEY = "tensorizer.notifyJobs";
const notifySupported = () => typeof window !== "undefined" && "Notification" in window;
function notifyStored(): boolean {
  try {
    return localStorage.getItem(NOTIFY_KEY) === "1" && notifySupported() && Notification.permission === "granted";
  } catch {
    return false;
  }
}
function storeNotify(on: boolean) {
  try {
    localStorage.setItem(NOTIFY_KEY, on ? "1" : "0");
  } catch {
    // storage blocked: the choice lasts for this page only
  }
}

const STATUS_STYLE: Record<JobStatus, string> = {
  queued: "bg-amber-50 text-amber-700",
  running: "bg-green-50 text-green-700",
  done: "bg-blue-50 text-blue-700",
  failed: "bg-red-50 text-red-700",
  cancelled: "bg-gray-100 text-gray-600",
};

/** 42 s → "0:42", 3 723 s → "1:02:03" */
function clock(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  const [h, m, r] = [Math.floor(s / 3600), Math.floor((s % 3600) / 60), s % 60];
  const pad = (n: number) => String(n).padStart(2, "0");
  return h ? `${h}:${pad(m)}:${pad(r)}` : `${m}:${pad(r)}`;
}

/** Top-bar Jobs indicator: a badge with the visitor's queued and running jobs, opening the list of their runs
 *  (they keep going on the server after the page closes). Optionally a browser notification when one finishes. */
export default function JobsButton({ titleOf }: { titleOf: (problem: string) => string }) {
  const { t } = useLang();
  const [jobs, setJobs] = useState<JobInfo[] | null>(null);
  const [open, setOpen] = useState(false);
  const [notify, setNotify] = useState(notifyStored);
  const [blocked, setBlocked] = useState(false);
  const [now, setNow] = useState(() => Date.now() / 1000);
  const previous = useRef<Map<string, JobStatus> | null>(null);
  const notifyRef = useRef(notify);
  notifyRef.current = notify;
  const titleRef = useRef(titleOf);
  titleRef.current = titleOf;
  const tRef = useRef(t); // the language may change while the poll keeps running
  tRef.current = t;
  const active = jobs?.filter(jobActive).length ?? 0;

  async function load() {
    const list = await listJobs();
    setNow(Date.now() / 1000);
    if (!list) return setJobs((j) => j ?? null);
    // A job that was queued or running on the last look and has ended since: tell the visitor if they asked.
    const before = previous.current;
    if (before && notifyRef.current && notifySupported() && Notification.permission === "granted") {
      for (const job of list) {
        const was = before.get(job.id);
        if (was && (was === "queued" || was === "running") && !jobActive(job) && job.status !== "cancelled") {
          const body =
            job.status === "done"
              ? tRef.current("jobs.notifyDone")
              : tRef.current("jobs.notifyFailed").replace("{error}", job.error ?? "");
          try {
            new Notification(titleRef.current(job.problem), { body, tag: `tensorizer-job-${job.id}` });
          } catch {
            // some browsers only allow notifications from a service worker
          }
        }
      }
    }
    previous.current = new Map(list.map((j) => [j.id, j.status]));
    setJobs(list);
  }

  useEffect(() => {
    void load();
    const onChange = () => void load();
    window.addEventListener(JOBS_CHANGED, onChange);
    return () => window.removeEventListener(JOBS_CHANGED, onChange);
  }, []);

  // Look again every few seconds while something runs or the list is open; the clock ticks while it is open.
  useEffect(() => {
    if (!active && !open) return;
    const poll = window.setInterval(() => void load(), 3000);
    const tick = open ? window.setInterval(() => setNow(Date.now() / 1000), 1000) : undefined;
    return () => {
      window.clearInterval(poll);
      window.clearInterval(tick);
    };
  }, [active > 0, open]);

  useEffect(() => {
    if (!open) return;
    const key = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [open]);

  async function toggleNotify() {
    if (notify) {
      setNotify(false);
      storeNotify(false);
      return;
    }
    if (!notifySupported()) return;
    // The browser asks for permission only now, on the visitor's own click.
    const permission =
      Notification.permission === "default" ? await Notification.requestPermission() : Notification.permission;
    setBlocked(permission === "denied");
    setNotify(permission === "granted");
    storeNotify(permission === "granted");
  }

  return (
    <>
      <button
        onClick={() => {
          setOpen(true);
          void load();
        }}
        title={t("jobs.buttonHint")}
        data-testid="jobs-button"
        className="relative flex h-8 items-center gap-1.5 rounded-md border border-blue-400 bg-blue-600 px-2.5 text-xs text-white hover:bg-blue-500"
      >
        <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
          <rect x="2" y="2.5" width="12" height="11" rx="1.5" />
          <path d="M5 6.5h6M5 9.5h4" strokeLinecap="round" />
        </svg>
        <span className="hidden sm:inline">{t("jobs.button")}</span>
        {active > 0 && (
          <span
            className="min-w-4 rounded-full bg-amber-400 px-1 text-center text-[10px] font-semibold leading-4 text-blue-900"
            data-testid="jobs-badge"
          >
            {active}
          </span>
        )}
      </button>
      {open && (
        <div
          className="fixed inset-0 z-50 flex items-start justify-center bg-gray-900/40 p-4 text-left text-gray-900"
          onMouseDown={() => setOpen(false)}
        >
          <div
            role="dialog"
            aria-label={t("jobs.title")}
            data-testid="jobs-panel"
            onMouseDown={(e) => e.stopPropagation()}
            className="mt-16 w-full max-w-md overflow-hidden rounded-xl bg-white shadow-2xl"
          >
            <div className="flex items-center justify-between border-b border-gray-200 px-4 py-2.5">
              <h2 className="text-sm font-semibold">{t("jobs.title")}</h2>
              <button
                onClick={() => setOpen(false)}
                aria-label={t("plot.close")}
                className="text-xl leading-none text-gray-400 hover:text-gray-700"
              >
                ×
              </button>
            </div>
            <div className="space-y-3 px-4 py-3 text-sm">
              <p className="text-xs text-gray-500">{t("jobs.hint")}</p>
              {jobs === null ? (
                <p className="text-amber-700">{t("jobs.offline")}</p>
              ) : jobs.length === 0 ? (
                <p className="text-gray-500">{t("jobs.empty")}</p>
              ) : (
                <ul className="max-h-[60vh] divide-y divide-gray-100 overflow-y-auto rounded-lg border border-gray-200">
                  {jobs.map((job) => (
                    <li key={job.id} className="flex items-center gap-2 px-3 py-1.5 text-xs" data-testid="job-row">
                      <span className={`shrink-0 rounded-full px-2 py-0.5 font-medium ${STATUS_STYLE[job.status]}`}>
                        {t(`jobs.${job.status}` as Key)}
                        {job.status === "queued" && job.position ? ` #${job.position}` : ""}
                      </span>
                      <button
                        onClick={() => {
                          requestOpenJob(job);
                          window.location.hash = `/${job.problem}`;
                          setOpen(false);
                        }}
                        title={job.error ?? t("jobs.open")}
                        className="min-w-0 flex-1 truncate text-left text-gray-800 hover:text-blue-700 hover:underline"
                      >
                        {titleOf(job.problem)}
                      </button>
                      <span className="w-14 shrink-0 text-right text-gray-500 tabular-nums">
                        {clock((job.finished ?? now) - job.created)}
                      </span>
                      {jobActive(job) && (
                        <button
                          onClick={async () => {
                            await cancelJob(job.id);
                            void load();
                          }}
                          className="shrink-0 rounded-md border border-red-200 px-2 py-0.5 text-red-700 hover:bg-red-50"
                        >
                          {t("jobs.cancel")}
                        </button>
                      )}
                    </li>
                  ))}
                </ul>
              )}
              {notifySupported() && (
                <label className="flex cursor-pointer items-center gap-2 text-xs text-gray-600">
                  <input type="checkbox" checked={notify} onChange={() => void toggleNotify()} />
                  {t("jobs.notify")}
                </label>
              )}
              {blocked && <p className="text-[11px] text-amber-700">{t("jobs.notifyBlocked")}</p>}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
