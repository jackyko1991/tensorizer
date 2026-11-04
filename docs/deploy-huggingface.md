# Deploying the backend on Hugging Face (free)

> **Optional, for self-hosters.** Tensorizer no longer runs a shared demo server: the website runs problems on
> **This device** or on **your own server** (⚙ Backend → Remote, see [remote-server.md](remote-server.md)). The Space
> and its deploy workflow still work if you want your own runner on Hugging Face (a private Space or HF PRO hardware
> suits this best). Add it in the site as one of your servers. For a private Space, set `TENSORIZER_SHARED=0` and
> a `TENSORIZER_TOKEN` *secret* so it needs your pairing token; a public Space should keep `TENSORIZER_SHARED=1`.

The website stays on GitHub Pages. This sets up the API that re-renders problems with your parameters
on a free Hugging Face **Docker Space** (2 vCPU, 16 GB RAM). It sleeps after about 48 hours without
visitors, and the first request afterwards takes about a minute to wake it.

Once it is set up, every push to `main` that changes problems or the backend redeploys it automatically.

## 1. Create a Hugging Face account

1. Go to <https://huggingface.co/join> and sign up. Use the username you want in the URL, for example `jackyko1991`.
2. Confirm the e-mail address. Spaces stay disabled until you do.

## 2. Create the Space

1. Open <https://huggingface.co/new-space>.
2. Fill in the form:
   - **Owner**: your username.
   - **Space name**: `tensorizer-api`.
   - **License**: apache-2.0.
   - **Select the Space SDK**: **Docker**, then **Blank**.
   - **Space hardware**: **CPU basic · 2 vCPU · 16 GB · FREE**.
   - **Visibility**: **Public**. A private Space can't be called from the website.
3. Click **Create Space**. Leave it empty; GitHub fills it in step 5.

The API's address is then `https://<username>-tensorizer-api.hf.space`, for example
`https://jackyko1991-tensorizer-api.hf.space`.

## 3. Create an access token

1. Open <https://huggingface.co/settings/tokens> and click **Create new token**.
2. Choose **Write**, name it `github-deploy`, and click **Create token**.
3. Copy the token, which starts with `hf_`. It is shown only once.

## 4. Tell GitHub about the Space

In the GitHub repository, go to **Settings → Secrets and variables → Actions**.

- **Secrets** tab → **New repository secret**: name `HF_TOKEN`, value the token from step 3.
- **Variables** tab → **New repository variable**: add both of these.
  - `HF_SPACE` = `<username>/tensorizer-api`.
  - `TENSORIZER_API_URL` = `https://<username>-tensorizer-api.hf.space`.

## 5. Deploy

1. Go to **Actions → Deploy backend to Hugging Face Space → Run workflow**. The workflow also runs on every push that changes the backend.
2. Watch the Space's **Logs** tab on Hugging Face. The first build installs PyTorch and takes about 10 minutes.
3. When it shows **Running**, open `https://<username>-tensorizer-api.hf.space/api/health`. You should see `{"status":"ok", ...}`.
4. Re-run **Actions → Deploy site to GitHub Pages** so the website picks up `TENSORIZER_API_URL`.

On <https://jackyko1991.github.io/tensorizer/>, the badge in the top bar now says **backend online**, and **Run** works.

## Settings and limits

These are set in the `Dockerfile`. To change one, add a Space variable with the same name (**Space → Settings → Variables**):

| Variable | Default | Meaning |
|---|---|---|
| `TENSORIZER_MAX_CONCURRENT` | 1 | Runs at the same time. On the 2 free CPUs, one at a time finishes the line fastest. |
| `TENSORIZER_MAX_QUEUE` | 20 | Waiting runs before new ones are refused with "server is busy". |
| `TENSORIZER_PER_CLIENT` | 1 | Runs one visitor (IP address) may have queued or running. |
| `TENSORIZER_JOB_TIMEOUT_S` | 600 | Wall-clock limit (s) of a run started as a job, which keeps going after the page closes. |
| `TENSORIZER_JOB_TTL_S` | 86400 | How long (s) finished jobs and their output stay listed. The Space's disk is temporary, so a restart clears them anyway. |
| `CORS_ORIGINS` | `https://jackyko1991.github.io` | Websites allowed to call the API. |
| `AGENT_ENABLED` | 1 | The Ask Claude panel. Visitors bring their own Anthropic API key, and Claude Code runs per chat in a scratch folder. |
| `MAX_AGENT_SESSIONS` | 2 | Chats answered at the same time (each runs one Claude Code process, about 1 GB RAM). |
| `ALLOW_CODE_EXEC` | off | Keep it off on a public Space. It allows running edited code, which is arbitrary Python. With `TENSORIZER_SHARED=1` it also needs `TENSORIZER_SHARED_CODE=1`. |
| `TENSORIZER_TOKEN` (secret) | none | Pairing token: every route except `/api/health` then needs `Authorization: Bearer <token>` (or set `TENSORIZER_TOKEN_SHA256`, its hex SHA-256). With a token and `TENSORIZER_SHARED=0`, edited code, the Workspace and the agent are on unless set to 0. |
| `TENSORIZER_SHARED` | 1 | Public-server defences. Set 0 only together with a token, for a Space only you use. |

The agent can still run code it wrote through its `run_problem` tool. That code runs inside the disposable Space container, with a timeout, through the same queue, and with no secrets in its environment. The visitor's API key exists only in that chat's Claude Code process and is never stored or logged.

While a run waits, each visitor sees their place in the queue in the terminal, for example "waiting in the queue — position 3".
Results are cached, so a run someone else already did with the same parameters returns instantly without waiting.

## Running on your own computer instead

Visitors can pick **This computer** in the top-bar menu to send runs to a backend on their own machine. It uses
their own CPU or GPU, has no queue, and allows code editing. To set it up:

```bash
git clone https://github.com/jackyko1991/tensorizer && cd tensorizer
pixi run api          # http://localhost:8000
```

The public site may call `localhost` because the API answers the browser's Private Network Access check. Chrome
may ask once for permission to access devices on your local network.
