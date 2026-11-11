/** The account: real (learnkit-hub) when a hub is configured, otherwise the demo account.
 *
 *  Hub (VITE_HUB_URL at build time, or `tensorizer.hubUrl` in localStorage for a local demo): sign-in by email link,
 *  the plan comes from the hub's signed token (Free: one Workspace problem; Pro: more). No hub (community mode): the
 *  demo sign-in only stores a display name in localStorage, nothing leaves the browser; not signed in is the free
 *  tier (one Workspace problem), the demo member tier has no limit. */
import { useEffect, useState } from "react";
import { grantFromHash, HubClient } from "./hub.ts";

export type Tier = "free" | "member" | "pro";
export interface Account {
  name: string;
  tier: Tier;
  since: number;
  /** a real account (hub): its email, and the Workspace limit its plan allows */
  email?: string;
  limit?: number;
}

/** The hub's address, or "" (community mode). */
export function hubUrl(storage: KeyValueLike | null = browserStorageLike()): string {
  const env = (import.meta as { env?: Record<string, string | undefined> }).env?.VITE_HUB_URL ?? "";
  if (env) return env;
  try {
    return storage?.getItem("tensorizer.hubUrl") ?? "";
  } catch {
    return "";
  }
}
type KeyValueLike = Pick<Storage, "getItem">;
const browserStorageLike = (): KeyValueLike | null => {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
};
export const hub: HubClient | null = hubUrl() ? new HubClient({ url: hubUrl(), product: "tensorizer" }) : null;

const RETURN_ROUTE = "tensorizer.signInReturn";
/** The address a sign-in link should come back to (the hub drops the #route; it is kept here). */
export function signInReturnTo(): string {
  try {
    localStorage.setItem(RETURN_ROUTE, location.hash);
  } catch {
    // the page reopens at the start instead
  }
  return location.origin + location.pathname + location.search;
}
/** On load: a sign-in link brought us back with #lk_grant=…: restore the route at once (before the router reads
 *  it), then swap the grant for a session. */
export function finishSignIn(): Promise<void> | null {
  const found = hub ? grantFromHash(location.hash) : null;
  if (!hub || !found) return null;
  let route = found.rest;
  try {
    route = localStorage.getItem(RETURN_ROUTE) || route;
    localStorage.removeItem(RETURN_ROUTE);
  } catch {
    // keep what the address had
  }
  history.replaceState(null, "", location.pathname + location.search + route);
  return hub.completeSignIn(found.grant).then(
    () => undefined,
    (e: Error) => alert(e.message),
  );
}

const KEY = "tensorizer.account";
const CHANGED = "tensorizer:account-changed";
/** Workspace problems allowed per tier. */
export const WORKSPACE_LIMIT: Record<Tier, number> = { free: 1, member: Number.POSITIVE_INFINITY, pro: 50 };

type KeyValue = Pick<Storage, "getItem" | "setItem" | "removeItem">;
const browserStorage = (): KeyValue | null => {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
};

/** The signed-in account (hub or demo), or null: the free tier. */
export function currentAccount(storage: KeyValue | null = browserStorage()): Account | null {
  if (hub && storage === browserStorage()) {
    const s = hub.session();
    if (!s) return null;
    const e = hub.entitlement();
    return { name: s.user.name || s.user.email, email: s.user.email, tier: e.tier === "pro" ? "pro" : "free", since: 0, limit: e.workspace_problems };
  }
  try {
    const raw = storage?.getItem(KEY);
    const data = raw ? JSON.parse(raw) : null;
    return data && typeof data.name === "string" && data.name.trim()
      ? { name: data.name.trim().slice(0, 40), tier: "member", since: Number(data.since) || Date.now() }
      : null;
  } catch {
    return null;
  }
}

export const tierOf = (account: Account | null): Tier => account?.tier ?? "free";

/** How many more Workspace problems this tier may add (Infinity for members). */
export function workspaceRoom(count: number, account: Account | null = currentAccount()): number {
  return Math.max(0, (account?.limit ?? WORKSPACE_LIMIT[tierOf(account)]) - count);
}

const changed = () => globalThis.dispatchEvent?.(new Event(CHANGED));

export function signIn(name: string, storage: KeyValue | null = browserStorage()): Account | null {
  const clean = name.trim().slice(0, 40);
  if (!clean) return null;
  const account: Account = { name: clean, tier: "member", since: Date.now() };
  try {
    storage?.setItem(KEY, JSON.stringify(account));
  } catch {
    return null; // storage blocked: stay on the free tier
  }
  changed();
  return account;
}

export function signOut(storage: KeyValue | null = browserStorage()) {
  if (hub && storage === browserStorage()) {
    void hub.signOut();
    return;
  }
  try {
    storage?.removeItem(KEY);
  } catch {
    // nothing stored
  }
  changed();
}

/** The current (demo) account, updated when it signs in or out (in any part of the page). */
export function useAccount(): Account | null {
  const [account, setAccount] = useState(() => currentAccount());
  useEffect(() => {
    const update = () => setAccount(currentAccount());
    window.addEventListener(CHANGED, update);
    window.addEventListener("storage", update); // another tab signed in or out
    const unsubscribe = hub?.subscribe(update); // hub sign-in, refresh (a new plan) or sign-out
    return () => (window.removeEventListener(CHANGED, update), window.removeEventListener("storage", update), unsubscribe?.());
  }, []);
  return account;
}
