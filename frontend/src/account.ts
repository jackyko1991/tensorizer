/** A mock account, to simulate signed-in and free-tier behaviour before real accounts exist.
 *
 *  Nothing leaves the browser: "Sign in (demo)" only stores a display name in localStorage. Not signed in is the
 *  free tier, which allows one Workspace problem; the demo member tier has no limit. Real sign-in later replaces
 *  `signIn` and the stored record, and the limits move to the server. */
import { useEffect, useState } from "react";

export type Tier = "free" | "member";
export interface Account {
  name: string;
  tier: Tier;
  since: number;
}

const KEY = "tensorizer.account";
const CHANGED = "tensorizer:account-changed";
/** Workspace problems allowed per tier. */
export const WORKSPACE_LIMIT: Record<Tier, number> = { free: 1, member: Number.POSITIVE_INFINITY };

type KeyValue = Pick<Storage, "getItem" | "setItem" | "removeItem">;
const browserStorage = (): KeyValue | null => {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
};

/** The signed-in (demo) account, or null: the free tier. */
export function currentAccount(storage: KeyValue | null = browserStorage()): Account | null {
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
  return Math.max(0, WORKSPACE_LIMIT[tierOf(account)] - count);
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
    return () => (window.removeEventListener(CHANGED, update), window.removeEventListener("storage", update));
  }, []);
  return account;
}
