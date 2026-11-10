// The demo account: not signed in = free tier (one Workspace problem), signed in = no limit. Run: node --test frontend/tests
import assert from "node:assert/strict";
import { test } from "node:test";
import { currentAccount, signIn, signOut, tierOf, workspaceRoom } from "../src/account.ts";

function memory() {
  const data = new Map<string, string>();
  return {
    getItem: (k: string) => data.get(k) ?? null,
    setItem: (k: string, v: string) => void data.set(k, v),
    removeItem: (k: string) => void data.delete(k),
    data,
  };
}

test("the free tier allows one Workspace problem; a demo sign-in lifts it", () => {
  const store = memory();
  assert.equal(currentAccount(store), null);
  assert.equal(tierOf(null), "free");
  assert.equal(workspaceRoom(0, null), 1);
  assert.equal(workspaceRoom(1, null), 0);
  const account = signIn("  Ada  ", store);
  assert.equal(account?.name, "Ada");
  assert.equal(currentAccount(store)?.tier, "member");
  assert.equal(workspaceRoom(5, currentAccount(store)), Number.POSITIVE_INFINITY);
  assert.equal(signIn("   ", store), null); // no name: nothing changes
  signOut(store);
  assert.equal(currentAccount(store), null);
  store.data.set("tensorizer.account", "not json");
  assert.equal(currentAccount(store), null);
});
