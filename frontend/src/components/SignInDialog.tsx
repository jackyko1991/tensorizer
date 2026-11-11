import { useEffect, useState } from "react";
import { hub, signInReturnTo } from "../account";
import { useLang, type Key } from "../i18n";

type Step = "email" | "sent" | "waitlist" | "joined" | "closed" | "disabled";
const BACKGROUNDS = ["student", "engineer", "researcher", "hobbyist"] as const;
const PRODUCTS = [
  ["tensorizer", "Tensorizer"],
  ["formalizer", "Formalizer"],
] as const;

/** Sign in by email link (learnkit-hub), or join the waitlist while accounts are not open yet.
 *  `waitlist` opens straight on the waitlist form (the roadmap's "Join the waitlist"). */
export default function SignInDialog({ onClose, waitlist = false }: { onClose: () => void; waitlist?: boolean }) {
  const { t } = useLang();
  const [step, setStep] = useState<Step>(waitlist ? "waitlist" : "email");
  const [email, setEmail] = useState("");
  const [mock, setMock] = useState(false);
  const [background, setBackground] = useState("");
  const [products, setProducts] = useState<string[]>(["tensorizer"]);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const key = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [onClose]);
  if (!hub) return null;
  const client = hub;

  const act = async (fn: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const send = () =>
    act(async () => {
      const r = await client.start(email.trim(), signInReturnTo());
      if (r.status === "sent") {
        setMock(r.mail === "mock");
        setStep("sent");
      } else setStep(r.status);
    });
  const join = () =>
    act(async () => {
      await client.joinWaitlist({ email: email.trim(), products, background, note: note.trim(), source: location.hash || "/" });
      setStep("joined");
    });

  const input = "w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-400 focus:outline-none";
  const primary = "rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50";
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-black/30 px-4 pt-20" onClick={onClose}>
      <div
        role="dialog"
        aria-label={t(step === "waitlist" || step === "joined" ? "acct.waitlistTitle" : "acct.title")}
        data-testid="sign-in-dialog"
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-md space-y-4 rounded-xl bg-white p-6 shadow-2xl"
      >
        <div className="flex items-start gap-3">
          <h2 className="text-lg font-semibold text-gray-900">
            {t(step === "waitlist" || step === "joined" ? "acct.waitlistTitle" : "acct.title")}
          </h2>
          <button onClick={onClose} aria-label={t("acct.close")} className="ml-auto rounded-md p-1 text-gray-500 hover:bg-gray-100">
            ✕
          </button>
        </div>

        {(step === "email" || step === "waitlist") && (
          <form
            className="space-y-3"
            onSubmit={(e) => {
              e.preventDefault();
              void (step === "email" ? send() : join());
            }}
          >
            <p className="text-sm text-gray-600">{t(step === "email" ? "acct.intro" : "acct.waitlistIntro")}</p>
            <label className="block text-sm">
              <span className="mb-1 block font-medium text-gray-700">{t("acct.email")}</span>
              <input
                type="email"
                required
                autoComplete="email"
                autoFocus
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className={input}
                data-testid="acct-email"
              />
            </label>
            {step === "waitlist" && (
              <>
                <fieldset>
                  <legend className="mb-1 text-sm font-medium text-gray-700">{t("acct.products")}</legend>
                  <div className="flex flex-wrap gap-2">
                    {PRODUCTS.map(([id, name]) => (
                      <label key={id} className="flex items-center gap-1.5 text-sm text-gray-700">
                        <input
                          type="checkbox"
                          checked={products.includes(id)}
                          onChange={(e) => setProducts((p) => (e.target.checked ? [...p, id] : p.filter((x) => x !== id)))}
                        />
                        {name}
                      </label>
                    ))}
                  </div>
                </fieldset>
                <fieldset>
                  <legend className="mb-1 text-sm font-medium text-gray-700">{t("acct.background")}</legend>
                  <div className="flex flex-wrap gap-1.5">
                    {BACKGROUNDS.map((b) => (
                      <button
                        key={b}
                        type="button"
                        aria-pressed={background === b}
                        onClick={() => setBackground((x) => (x === b ? "" : b))}
                        className={`rounded-full border px-3 py-1 text-xs ${
                          background === b ? "border-blue-600 bg-blue-600 text-white" : "border-gray-300 text-gray-700 hover:border-blue-300"
                        }`}
                      >
                        {t(`acct.bg.${b}` as Key)}
                      </button>
                    ))}
                  </div>
                </fieldset>
                <textarea
                  value={note}
                  maxLength={500}
                  rows={2}
                  onChange={(e) => setNote(e.target.value)}
                  placeholder={t("acct.note")}
                  aria-label={t("acct.note")}
                  className={input}
                />
              </>
            )}
            {error && <p className="text-sm text-red-700">{error}</p>}
            <button type="submit" disabled={busy || !email.trim() || (step === "waitlist" && !products.length)} className={primary}>
              {t(step === "email" ? "acct.send" : "acct.join")}
            </button>
          </form>
        )}

        {step === "sent" && (
          <div className="space-y-2 text-sm text-gray-700" data-testid="acct-sent">
            <p>{t("acct.sent")}</p>
            {mock && (
              <p className="rounded-md bg-amber-50 px-3 py-2 text-amber-900">
                {t("acct.mock")}{" "}
                <a href={`${client.url}/dev/outbox`} target="_blank" rel="noreferrer" className="font-medium underline">
                  {t("acct.outbox")} ↗
                </a>
              </p>
            )}
          </div>
        )}
        {step === "joined" && <p className="text-sm text-gray-700" data-testid="acct-joined">{t("acct.joined")}</p>}
        {step === "closed" && <p className="text-sm text-gray-700">{t("acct.closed")}</p>}
        {step === "disabled" && <p className="text-sm text-red-700">{t("acct.disabled")}</p>}
      </div>
    </div>
  );
}
