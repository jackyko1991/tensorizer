// Copy of learnkit/web/src/accounts.ts (copy first, adopt later): keep the two in step until Tensorizer uses learnkit.
/**
 * Accounts client for learnkit-hub (sign-in by email link, waitlist, plans). No business logic: the hub decides
 * who gets what and signs it into a short-lived token; this file only stores the tokens, refreshes them and reads
 * what the token says the user may do. Without a hub URL a site stays in community mode (no accounts).
 *
 * Sign-in: start(email) → the hub emails a link (mock mode: the hub's terminal and /dev/outbox) → the link opens
 * the hub, which redirects back to the site with #lk_grant=… in the address → completeSignIn() swaps it for tokens.
 *
 * Pure (fetch and storage are passed in), unit-tested in web/tests/accounts.test.ts.
 */

export type Tier = "free" | "pro";
export interface Entitlement {
  tier: Tier;
  workspace_problems: number;
  storage_mb: number;
  cloud_agent: boolean;
  remote_compute: string | null;
}
export interface HubUser {
  id: number;
  email: string;
  name: string;
  role: "user" | "admin";
}
export interface Session {
  access: string;
  refresh: string;
  /** ms since epoch when the access token expires */
  expires: number;
  user: HubUser;
}
export type Registration = Record<string, "closed" | "waitlist" | "open">;
/** what start() says: a link was sent, or why not (the product only takes waitlist sign-ups, is closed, …) */
export type StartResult = { status: "sent"; mail: string } | { status: "waitlist" | "closed" | "disabled" };

export const FREE: Entitlement = { tier: "free", workspace_problems: 1, storage_mb: 0, cloud_agent: false, remote_compute: null };

type KeyValue = Pick<Storage, "getItem" | "setItem" | "removeItem">;
type Fetch = (url: string, init?: RequestInit) => Promise<Response>;

export class HubError extends Error {
  status: number;
  constructor(message: string, status = 0) {
    super(message);
    this.status = status;
  }
}

/** The claims part of a JWT (not verified: the hub checks every call; this only drives the interface). */
export function tokenClaims(token: string): Record<string, unknown> | null {
  try {
    const part = token.split(".")[1];
    const json = atob(part.replace(/-/g, "+").replace(/_/g, "/").padEnd(Math.ceil(part.length / 4) * 4, "="));
    return JSON.parse(decodeURIComponent(escape(json)));
  } catch {
    return null;
  }
}

/** The grant the hub put in the address after a sign-in link, and the address without it. */
export function grantFromHash(hash: string): { grant: string; rest: string } | null {
  const m = hash.match(/(^#|&)lk_grant=([^&]+)/);
  if (!m) return null;
  const rest = hash.replace(/(^#|&)lk_grant=[^&]+/, "").replace(/^#&/, "#");
  return { grant: decodeURIComponent(m[2]), rest: rest === "#" ? "" : rest };
}

export interface HubClientOptions {
  /** the hub's address, e.g. http://localhost:8800 */
  url: string;
  /** this site's product id at the hub, e.g. "tensorizer" */
  product: string;
  storage?: KeyValue | null;
  fetch?: Fetch;
  /** storage key for the session (default "<product>.hubSession") */
  key?: string;
  now?: () => number;
}

export class HubClient {
  readonly url: string;
  readonly product: string;
  private storage: KeyValue | null;
  private fetcher: Fetch;
  private key: string;
  private now: () => number;
  private refreshing: Promise<Session | null> | null = null;
  private listeners = new Set<() => void>();

  constructor(o: HubClientOptions) {
    this.url = o.url.replace(/\/+$/, "");
    this.product = o.product;
    this.storage = o.storage === undefined ? safeStorage() : o.storage;
    this.fetcher = o.fetch ?? ((u, i) => fetch(u, i));
    this.key = o.key ?? `${o.product}.hubSession`;
    this.now = o.now ?? Date.now;
  }

  /** Called whenever the session changes (sign in, refresh, sign out). Returns an unsubscribe function. */
  subscribe(fn: () => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  session(): Session | null {
    try {
      const raw = this.storage?.getItem(this.key);
      const s = raw ? (JSON.parse(raw) as Session) : null;
      return s && s.access && s.refresh && s.user ? s : null;
    } catch {
      return null;
    }
  }

  /** What the current token allows for this product (FREE when signed out or not covered). */
  entitlement(): Entitlement {
    const s = this.session();
    const ent = s ? (tokenClaims(s.access)?.ent as Record<string, Entitlement> | undefined) : undefined;
    return { ...FREE, ...(ent?.[this.product] ?? {}) };
  }

  private save(s: Session | null) {
    try {
      if (s) this.storage?.setItem(this.key, JSON.stringify(s));
      else this.storage?.removeItem(this.key);
    } catch {
      // storage blocked: the session lasts until the page is left
    }
    this.listeners.forEach((fn) => fn());
  }

  private async call<T>(path: string, init: RequestInit = {}, auth = false): Promise<T> {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (auth) {
      const token = await this.accessToken();
      if (!token) throw new HubError("Sign in first.", 401);
      headers.Authorization = `Bearer ${token}`;
    }
    let res: Response;
    try {
      res = await this.fetcher(this.url + path, { ...init, headers: { ...headers, ...(init.headers as object) } });
    } catch {
      throw new HubError("The account service can't be reached.", 0);
    }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new HubError(typeof data.detail === "string" ? data.detail : `request failed (${res.status})`, res.status);
    return data as T;
  }

  registration(): Promise<Registration> {
    return this.call<Registration>("/api/registration");
  }

  joinWaitlist(entry: { email: string; products?: string[]; background?: string; note?: string; source?: string }) {
    return this.call<{ ok: boolean }>("/api/waitlist", {
      method: "POST",
      body: JSON.stringify({ products: [this.product], ...entry }),
    });
  }

  /** Ask for a sign-in link; the link brings the user back to returnTo (an address of this site). */
  start(email: string, returnTo: string): Promise<StartResult> {
    return this.call<StartResult>("/api/auth/email/start", {
      method: "POST",
      body: JSON.stringify({ email, product: this.product, return_to: returnTo }),
    });
  }

  private accept(data: { access: string; refresh: string; expires_in: number; user: HubUser }): Session {
    const s: Session = { access: data.access, refresh: data.refresh, expires: this.now() + data.expires_in * 1000, user: data.user };
    this.save(s);
    return s;
  }

  /** Swap the grant from a sign-in link for a session. */
  async completeSignIn(grant: string): Promise<Session> {
    return this.accept(await this.call("/api/auth/token", { method: "POST", body: JSON.stringify({ grant }) }));
  }

  /** A valid access token, refreshed when it is about to expire; null when signed out (or the refresh failed). */
  async accessToken(): Promise<string | null> {
    const s = this.session();
    if (!s) return null;
    if (s.expires - this.now() > 30_000) return s.access;
    this.refreshing ??= this.refresh(s).finally(() => (this.refreshing = null));
    return (await this.refreshing)?.access ?? null;
  }

  private async refresh(s: Session): Promise<Session | null> {
    try {
      return this.accept(await this.call("/api/auth/refresh", { method: "POST", body: JSON.stringify({ refresh: s.refresh }) }));
    } catch (e) {
      if (e instanceof HubError && (e.status === 401 || e.status === 403)) this.save(null); // revoked or disabled
      return null;
    }
  }

  /** The user and plans as the hub sees them now (refreshes the token so its entitlements are current). */
  async me(): Promise<{ user: HubUser; plans: string[]; ent: Record<string, Entitlement> }> {
    return this.call("/api/me", {}, true);
  }

  /** Refresh the token now, so a plan just changed by an admin shows up without waiting for expiry. */
  async reload(): Promise<void> {
    const s = this.session();
    if (s) await this.refresh(s);
  }

  async signOut(): Promise<void> {
    const s = this.session();
    this.save(null);
    if (s) await this.call("/api/auth/logout", { method: "POST", body: JSON.stringify({ refresh: s.refresh }) }).catch(() => {});
  }
}

function safeStorage(): KeyValue | null {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}
