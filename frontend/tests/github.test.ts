// Export to GitHub with a mocked fetch (nothing is sent anywhere). Run: node --test frontend/tests
import assert from "node:assert/strict";
import { test } from "node:test";
import { GitHubError, pushToGitHub, toBase64, treeEntries, validRepoName } from "../src/github.ts";

type Call = { method: string; path: string; body: any; auth: string };

/** A tiny fake of the GitHub REST endpoints the export uses. */
function fakeGitHub(opts: { repoExists?: boolean; userStatus?: number; createStatus?: number } = {}) {
  const calls: Call[] = [];
  let blobs = 0;
  const reply = (status: number, data: unknown) => new Response(JSON.stringify(data), { status });
  const fetch = (async (url: string, init: RequestInit) => {
    const path = url.replace("https://api.github.com", "");
    const method = init.method ?? "GET";
    const body = init.body ? JSON.parse(String(init.body)) : undefined;
    calls.push({ method, path, body, auth: String((init.headers as Record<string, string>).Authorization) });
    if (path === "/user") return reply(opts.userStatus ?? 200, { login: "ada" });
    if (path === "/repos/ada/ws" && method === "GET")
      return opts.repoExists
        ? reply(200, { default_branch: "main", html_url: "https://github.com/ada/ws", permissions: { push: true } })
        : reply(404, { message: "Not Found" });
    if (path === "/user/repos")
      return reply(opts.createStatus ?? 201, { default_branch: "main", html_url: "https://github.com/ada/ws", message: "name already exists" });
    if (path === "/repos/ada/ws/git/ref/heads/main") return reply(200, { object: { sha: "head0" } });
    if (path === "/repos/ada/ws/git/commits/head0") return reply(200, { tree: { sha: "tree0" } });
    if (path === "/repos/ada/ws/git/blobs") return reply(201, { sha: `blob${++blobs}` });
    if (path === "/repos/ada/ws/git/trees") return reply(201, { sha: "tree1" });
    if (path === "/repos/ada/ws/git/commits") return reply(201, { sha: "commit1" });
    if (path === "/repos/ada/ws/git/refs/heads/main") return reply(200, { object: { sha: "commit1" } });
    return reply(500, { message: `unexpected ${method} ${path}` });
  }) as unknown as typeof globalThis.fetch;
  return { fetch, calls };
}

const files = {
  "README.md": new TextEncoder().encode("# ws\n"),
  "problems/workspace/a/main.py": new TextEncoder().encode("print(1)\n"),
  "problems/workspace/a/data/m.csv": new Uint8Array([116, 44, 120, 10]),
};

test("tree entries are regular blobs in path order", () => {
  assert.deepEqual(treeEntries({ "b.py": "2", "a/x.csv": "1" }), [
    { path: "a/x.csv", mode: "100644", type: "blob", sha: "1" },
    { path: "b.py", mode: "100644", type: "blob", sha: "2" },
  ]);
  assert.equal(toBase64(new TextEncoder().encode("hi")), "aGk=");
  assert.ok(validRepoName("tensorizer-ws") && !validRepoName("a b") && !validRepoName(".."));
});

test("a new repository is created private, then one commit holds every file", async () => {
  const gh = fakeGitHub();
  const steps: string[] = [];
  const url = await pushToGitHub({
    token: "tok",
    repo: "ws",
    private: true,
    files,
    message: "Export",
    fetch: gh.fetch,
    onProgress: (p) => steps.push(p.step),
  });
  assert.equal(url, "https://github.com/ada/ws");
  assert.ok(gh.calls.every((c) => c.auth === "Bearer tok"));
  const create = gh.calls.find((c) => c.path === "/user/repos")!;
  assert.deepEqual([create.body.name, create.body.private, create.body.auto_init], ["ws", true, true]);
  assert.equal(gh.calls.filter((c) => c.path.endsWith("/git/blobs")).length, 3);
  const tree = gh.calls.find((c) => c.path.endsWith("/git/trees"))!.body;
  assert.equal(tree.base_tree, "tree0");
  assert.deepEqual(tree.tree.map((e: { path: string }) => e.path), Object.keys(files).sort());
  const commit = gh.calls.find((c) => c.path === "/repos/ada/ws/git/commits" && c.method === "POST")!.body;
  assert.deepEqual([commit.tree, commit.parents], ["tree1", ["head0"]]);
  assert.deepEqual(gh.calls.at(-1), { method: "PATCH", path: "/repos/ada/ws/git/refs/heads/main", body: { sha: "commit1" }, auth: "Bearer tok" });
  assert.deepEqual([...new Set(steps)], ["user", "repo", "blobs", "commit"]);
});

test("an existing repository is pushed to without being created", async () => {
  const gh = fakeGitHub({ repoExists: true });
  await pushToGitHub({ token: "t", repo: "ws", private: true, files, message: "m", fetch: gh.fetch });
  assert.ok(!gh.calls.some((c) => c.path === "/user/repos"));
});

test("errors are clear: bad token, name taken", async () => {
  const bad = fakeGitHub({ userStatus: 401 });
  await assert.rejects(
    pushToGitHub({ token: "x", repo: "ws", private: true, files, message: "m", fetch: bad.fetch }),
    (e: unknown) => e instanceof GitHubError && e.code === "badToken",
  );
  const taken = fakeGitHub({ createStatus: 422 });
  await assert.rejects(
    pushToGitHub({ token: "x", repo: "ws", private: true, files, message: "m", fetch: taken.fetch }),
    (e: unknown) => e instanceof GitHubError && e.code === "nameTaken",
  );
  const noScope = fakeGitHub({ createStatus: 403 });
  await assert.rejects(
    pushToGitHub({ token: "x", repo: "ws", private: true, files, message: "m", fetch: noScope.fetch }),
    (e: unknown) => e instanceof GitHubError && e.code === "scopes",
  );
});
