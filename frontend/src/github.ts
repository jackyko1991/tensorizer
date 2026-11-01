/**
 * Export a workspace to GitHub from the browser, as one commit made with the Git Data API (blobs → tree → commit →
 * update ref): https://docs.github.com/en/rest/git. The token is the user's fine-grained personal access token; it is
 * sent only to api.github.com. Self-contained (no imports) so `node --test frontend/tests` can run it with a mocked
 * fetch.
 */

export const API = "https://api.github.com";
export const TOKEN_URL = "https://github.com/settings/personal-access-tokens/new";

export type GitHubErrorCode = "badToken" | "scopes" | "nameTaken" | "other";
export class GitHubError extends Error {
  code: GitHubErrorCode;
  constructor(code: GitHubErrorCode, message: string) {
    super(message);
    this.code = code;
  }
}

export type Progress =
  | { step: "user" }
  | { step: "repo" }
  | { step: "blobs"; done: number; total: number }
  | { step: "commit" };

export interface PushOptions {
  token: string;
  repo: string;
  private: boolean;
  files: Record<string, Uint8Array>;
  message: string;
  onProgress?: (p: Progress) => void;
  fetch?: typeof fetch;
}

/** A repository name GitHub accepts. */
export const validRepoName = (name: string) => /^[A-Za-z0-9._-]{1,100}$/.test(name) && !/^\.+$/.test(name);

export function toBase64(bytes: Uint8Array): string {
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

/** The tree for one commit: every file as a regular blob, in path order. */
export function treeEntries(shas: Record<string, string>) {
  return Object.keys(shas)
    .sort()
    .map((path) => ({ path, mode: "100644" as const, type: "blob" as const, sha: shas[path] }));
}

/** Push `files` to <user>/<repo> (created when missing) as one commit on its default branch; returns its URL. */
export async function pushToGitHub(opts: PushOptions): Promise<string> {
  const http = opts.fetch ?? fetch;
  const call = async (method: string, path: string, body?: unknown): Promise<{ status: number; data: any }> => {
    let res: Response;
    try {
      res = await http(`${API}${path}`, {
        method,
        headers: {
          Accept: "application/vnd.github+json",
          Authorization: `Bearer ${opts.token}`,
          "X-GitHub-Api-Version": "2022-11-28",
          ...(body ? { "Content-Type": "application/json" } : {}),
        },
        body: body ? JSON.stringify(body) : undefined,
      });
    } catch (e) {
      throw new GitHubError("other", e instanceof Error ? e.message : String(e));
    }
    const data = res.status === 204 ? null : await res.json().catch(() => null);
    return { status: res.status, data };
  };
  const fail = (r: { status: number; data: any }, what: string): never => {
    const message = `${what}: ${r.data?.message ?? r.status}`;
    if (r.status === 401) throw new GitHubError("badToken", message);
    if (r.status === 403 || (r.status === 404 && what !== "repository")) throw new GitHubError("scopes", message);
    throw new GitHubError("other", message);
  };

  opts.onProgress?.({ step: "user" });
  const user = await call("GET", "/user");
  if (user.status !== 200) fail(user, "user");
  const owner: string = user.data.login;

  opts.onProgress?.({ step: "repo" });
  let repo = await call("GET", `/repos/${owner}/${opts.repo}`);
  if (repo.status === 404) {
    repo = await call("POST", "/user/repos", {
      name: opts.repo,
      private: opts.private,
      auto_init: true, // the Git Data API needs a first commit to build on
      description: "A Tensorizer workspace",
    });
    if (repo.status === 422) throw new GitHubError("nameTaken", repo.data?.errors?.[0]?.message ?? repo.data?.message ?? "");
    if (repo.status !== 201) fail(repo, "create repository");
  } else if (repo.status !== 200) fail(repo, "repository");
  if (repo.data.permissions && !repo.data.permissions.push) throw new GitHubError("scopes", "no push access");
  const base = `/repos/${owner}/${opts.repo}`;
  const branch: string = repo.data.default_branch || "main";

  // The branch head (a new repository may need a moment; an empty one gets a first commit through the contents API).
  let head: string | null = null;
  for (let attempt = 0; attempt < 5 && !head; attempt++) {
    const ref = await call("GET", `${base}/git/ref/heads/${branch}`);
    if (ref.status === 200) head = ref.data.object.sha;
    else if (ref.status === 409 || (ref.status === 404 && attempt === 2)) {
      const init = await call("PUT", `${base}/contents/README.md`, {
        message: "Initial commit",
        content: toBase64(new TextEncoder().encode("# Tensorizer workspace\n")),
        branch,
      });
      if (init.status !== 201 && init.status !== 200) fail(init, "initial commit");
      head = init.data.commit.sha;
    } else if (ref.status !== 404) fail(ref, "branch");
    else await new Promise((r) => setTimeout(r, opts.fetch ? 0 : 800));
  }
  if (!head) throw new GitHubError("other", `branch ${branch} not found`);
  const parent = await call("GET", `${base}/git/commits/${head}`);
  if (parent.status !== 200) fail(parent, "commit");

  const paths = Object.keys(opts.files);
  const shas: Record<string, string> = {};
  let done = 0;
  opts.onProgress?.({ step: "blobs", done, total: paths.length });
  for (const path of paths) {
    const blob = await call("POST", `${base}/git/blobs`, { content: toBase64(opts.files[path]), encoding: "base64" });
    if (blob.status !== 201) fail(blob, `upload ${path}`);
    shas[path] = blob.data.sha;
    opts.onProgress?.({ step: "blobs", done: ++done, total: paths.length });
  }

  opts.onProgress?.({ step: "commit" });
  const tree = await call("POST", `${base}/git/trees`, { base_tree: parent.data.tree.sha, tree: treeEntries(shas) });
  if (tree.status !== 201) fail(tree, "tree");
  const commit = await call("POST", `${base}/git/commits`, { message: opts.message, tree: tree.data.sha, parents: [head] });
  if (commit.status !== 201) fail(commit, "commit");
  const ref = await call("PATCH", `${base}/git/refs/heads/${branch}`, { sha: commit.data.sha });
  if (ref.status !== 200) fail(ref, "update branch");
  return repo.data.html_url ?? `https://github.com/${owner}/${opts.repo}`;
}
