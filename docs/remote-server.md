# Run Tensorizer on your own server

The website (<https://jackyko1991.github.io/tensorizer/>) is a static page. When you press **Run**, it sends the
problem to a *runner*: a small program that runs the Python code and sends back the animation. This page shows how
to put a runner on a machine you control (a rented VPS, a mini-PC at home, a lab server or a Windows PC) and connect
the website to it.

There is no shared demo server: problems run either on **This device** (a runner on the computer you browse from)
or on a server of your own, set up as below. The website's **⚙ Backend → Remote** tab has the same set-up paths
in short, step by step.

You need: a Linux machine you can log in to with `ssh` (or Windows with WSL2, see the last section), and about
10 minutes. Everything below is copy-paste. Lines starting with `#` are explanations, not commands.

## How it fits together

```
your browser (https page) ──https──▶  your server ──▶ runner container (127.0.0.1:8765)
```

- The **runner** is one container image, `ghcr.io/jackyko1991/tensorizer-runner`, the same one the free
  Hugging Face demo uses. The installer runs it as a systemd service called `tensorizer-runner`, listening only on
  the server itself (`127.0.0.1:8765`), never directly on the internet.
- The website is an `https://` page, and browsers refuse to call a plain `http://` server from it, except for
  `localhost`. So the runner needs an **HTTPS address**. Pick the row that matches your machine:

| Your situation | HTTPS by | Installer option | Section |
|---|---|---|---|
| VPS (AWS, Azure, GCP, Hetzner, …) and you own a domain | Caddy, automatic certificate | `--domain runner.example.org` | [A](#a-vps-with-a-domain) |
| VPS without a domain | Caddy with a free `<ip>.sslip.io` name | `--sslip` | [B](#b-vps-without-a-domain-sslipio) |
| Home server (e.g. a Beelink mini-PC), private use | Tailscale | `--no-tls` | [C1](#c1-home-server-with-tailscale-private) |
| Home server, public URL | Cloudflare Tunnel | `--no-tls` | [C2](#c2-home-server-with-a-cloudflare-tunnel-public-url) |
| Machine behind a university/company VPN or jump host (also without root/Docker, e.g. HPC) | an SSH tunnel | `--no-tls`, or no installer | [D](#d-a-machine-behind-a-vpn-ssh-tunnel) |
| Windows PC | WSL2 | `--no-tls` | [E](#e-windows-with-wsl2) |
| Mac (Apple GPU) | Tailscale, or none on the same Mac | no installer (native) | [F](#f-mac-native-uses-the-apple-gpu) |
| ARM64: Raspberry Pi 4/5, AWS Graviton, Ampere, Oracle Always Free | Tailscale or Cloudflare Tunnel | `--no-tls` | [G](#g-arm64-device-raspberry-pi-45-cloud-arm-servers) |

**Architectures.** The CPU image `:latest` / `:<version>` is multi-arch: the same tag runs on x86_64 (amd64) and
ARM64 (arm64), and `docker pull` picks the right one. The CUDA image (`:cuda`, `:<version>-cuda`, `--gpu`) is
x86_64 only; the installer stops with a message if you ask for it on ARM.

## What the installer does

The script is [`scripts/install.sh`](../scripts/install.sh) in this repository; read it before you run it. In short:

1. installs Docker with Docker's official script (or uses Docker/Podman if already installed);
2. downloads the runner image;
3. writes the settings, including a random **pairing token**, to `/etc/tensorizer/runner.env` (readable by root only);
4. creates the `tensorizer-runner` service (and, for `--domain`/`--sslip`, a `tensorizer-caddy` service for HTTPS);
5. prints the URL and the token. **The token is shown only once**: copy it somewhere safe (a password manager).

To see exactly what it would do without changing anything, add `--dry-run`:

```sh
curl -fsSL https://jackyko1991.github.io/tensorizer/install.sh | sh -s -- --dry-run --sslip
```

Useful options (all of them: `--help`):

| Option | Meaning |
|---|---|
| `--gpu` | use the NVIDIA GPU (installs the `-cuda` image, x86_64 only; needs the NVIDIA driver and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)) |
| `--port 8765` | the runner's local port |
| `--memory 4g`, `--cpus 2` | cap what the runner may use |
| `--shared` | turn on the public-demo defences (time limits, one job per visitor, no edited code); see [Security](#security) |
| `--new-token` | make a new pairing token (the old one stops working) |
| `--version 0.1.0` | a specific runner version instead of `latest` |
| `--uninstall` | remove everything the installer made |

**Updating:** run the same install command again. It downloads the newest image, restarts the service and keeps
your token.

## Connect the website

1. Open <https://jackyko1991.github.io/tensorizer/> → **⚙ Backend** → **Remote** → **+ Add server**.
2. Give it a name, paste the URL and the token the installer printed, and **Save**. The site runs **Test
   connection** at once: a public health check, then one call with your token. You should see the runner version,
   the devices (`cpu`, or `cpu, cuda` with `--gpu`), the queue, and "private · runs your own code".
3. Press **Use**. Runs, jobs, the Workspace, notebooks and the AI assistant now go to your server.

Where the token is kept: only in your browser, in this tab's session storage (forgotten when you close the tab), or
in local storage if you tick **Remember on this device**. It is sent only to that server, in the
`Authorization: Bearer` header, never in a URL. Run pictures are loaded by `<img>` tags, which cannot send a header,
so the server signs each picture URL it hands out (`?sig=…`, valid for that one file, and only until the token
changes).

What Test connection's messages mean:

| Message | Cause and fix |
|---|---|
| browsers block plain http:// servers | the site is `https://`, so it may only call `https://` servers or `http://localhost`: use options A–C, or an SSH tunnel (D) |
| can't reach the server | the runner is not running, the address is wrong, or your VPN/tunnel is down; see [Troubleshooting](#troubleshooting) |
| doesn't allow this website (CORS) | the server's `CORS_ORIGINS` doesn't list this site (the installer sets `https://jackyko1991.github.io`) |
| wrong or missing pairing token | paste the token again (**Edit**); lost it? re-run the installer with `--new-token` |
| too many wrong tokens | after 10 wrong tokens in a minute the server waits a minute before answering again |

---

## A. VPS with a domain

You rent a server and own a domain (for example `example.org`), and want `https://runner.example.org`.

```sh
# 1. In your domain's DNS settings, add an "A" record:  runner  →  <your server's public IP address>
#    (check it after a few minutes:)
getent hosts runner.example.org

# 2. Open ports 80 and 443 in your cloud provider's firewall / security group (AWS: Security Groups,
#    Azure: Network security group, GCP: VPC firewall rules, Hetzner: Firewalls). Caddy needs both for the certificate.

# 3. Log in to the server and run the installer:
ssh you@runner.example.org
curl -fsSL https://jackyko1991.github.io/tensorizer/install.sh | sudo sh -s -- --domain runner.example.org
```

`curl … | sudo sh -s -- <options>` downloads the script and runs it as root with your options. When it finishes it
prints `URL : https://runner.example.org` and the token. Caddy renews the certificate by itself.

## B. VPS without a domain (sslip.io)

[sslip.io](https://sslip.io) is a free service that turns an IP address into a name: `203-0-113-7.sslip.io` points
at `203.0.113.7`. That is enough for Caddy to get a real certificate.

```sh
# 1. Open ports 80 and 443 in your cloud provider's firewall (see A, step 2).
# 2. On the server:
curl -fsSL https://jackyko1991.github.io/tensorizer/install.sh | sudo sh -s -- --sslip
```

The installer finds the server's public IP address by itself (`--ip 203.0.113.7` to choose it) and prints a URL
like `https://203-0-113-7.sslip.io`. If your provider changes the IP address after a restart, re-run the installer.

## C1. Home server with Tailscale (private)

[Tailscale](https://tailscale.com) joins your devices into a private network (a "tailnet") and gives each one an
HTTPS name such as `https://beelink.tail1234.ts.net`. Only devices logged in to your tailnet can reach it, and
nothing is opened on your home router. Free for personal use.

```sh
# 1. On the home server: install the runner, local only
curl -fsSL https://jackyko1991.github.io/tensorizer/install.sh | sudo sh -s -- --no-tls

# 2. Install Tailscale and log in (it prints a link to open in your browser)
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# 3. In the Tailscale admin console (https://login.tailscale.com/admin/dns): turn on MagicDNS and
#    "HTTPS Certificates".

# 4. Publish the runner on the tailnet over HTTPS (--bg keeps it running after you log out and after reboots)
sudo tailscale serve --bg 8765
tailscale serve status        # shows the https://<name>.<tailnet>.ts.net address
```

5. Install Tailscale on the computer you browse from too, and log in with the same account.
6. In the website, add the `https://….ts.net` address and the token.

## C2. Home server with a Cloudflare Tunnel (public URL)

A [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) gives the
home server a public HTTPS address without opening ports on your router. You need a free Cloudflare account and a
domain whose DNS is on Cloudflare.

```sh
# 1. On the home server: install the runner, local only
curl -fsSL https://jackyko1991.github.io/tensorizer/install.sh | sudo sh -s -- --no-tls
```

2. In the Cloudflare dashboard: **Zero Trust → Networks → Tunnels → Create a tunnel → Cloudflared**. Name it
   `tensorizer`.
3. The dashboard shows install commands for your system, ending with `sudo cloudflared service install <long-key>`.
   Copy and run them on the home server (they install `cloudflared` as a service that starts at boot).
4. Under **Public hostname**, add: subdomain `runner`, your domain, service type `HTTP`, URL `localhost:8765`.
   Save.
5. In the website, add `https://runner.<your-domain>` and the token.

Just trying it? `cloudflared tunnel --url http://127.0.0.1:8765` prints a temporary `https://….trycloudflare.com`
address with no account needed; it changes every time and stops when you close the terminal.

A public URL means anyone can find it: keep the token secret, and consider `--shared` (see [Security](#security)).

## D. A machine behind a VPN (SSH tunnel)

> **Check your institution's policy first.** Many universities, labs and companies do not allow installing software,
> running servers, tunnels or remote-access tools (including Tailscale or Cloudflare) on their machines or network
> without approval. Ask your IT or system administrators before you start, and follow their rules.

A university, lab or company server that you reach only through a VPN or a jump host (bastion). The website talks
to the runner **straight from your browser**, so this works only while your VPN or tunnel is up.

### D1. You have root and Docker (or Podman) on the machine

```sh
# 1. On the machine: install the runner, local only (add --gpu for NVIDIA GPUs)
curl -fsSL https://jackyko1991.github.io/tensorizer/install.sh | sudo sh -s -- --no-tls

# 2. On your own computer, with the VPN connected; keep this terminal open while you work:
ssh -N -L 8765:localhost:8765 you@lab-server.example.org
#    -L 8765:localhost:8765  forwards your port 8765 to port 8765 on the machine
#    -N                      no remote shell, just the tunnel
# Through a jump host / bastion:
ssh -N -J you@jumphost.example.org -L 8765:localhost:8765 you@lab-server.example.org
```

3. In the website: **⚙ Backend → Remote → + Add server**, URL `http://localhost:8765`, and the token. Browsers allow
   an `https://` page to call `http://localhost`, so no certificate is needed.

If something on your computer already uses port 8765 (for example a local runner), forward another one:
`ssh -N -L 8766:localhost:8765 …` and add `http://localhost:8766`.

Instead of a tunnel you can give the machine HTTPS inside the VPN: Tailscale on it (C1) if your institute allows
it, or a reverse proxy such as Caddy with an internal certificate from your IT department, pointing at
`127.0.0.1:8765`. Then add that `https://` address with the token.

### D2. No root or Docker (common on HPC and lab servers)

Run the runner natively in your home folder with [pixi](https://pixi.sh); nothing needs administrator rights.

```sh
# 1. On the machine: install pixi and get the code
curl -fsSL https://pixi.sh/install.sh | sh          # then open a new shell
git clone https://github.com/jackyko1991/tensorizer && cd tensorizer && pixi install

# 2. Make a token (copy it somewhere safe), then start the runner on this machine only (127.0.0.1)
export TENSORIZER_TOKEN=$(openssl rand -hex 32); echo "$TENSORIZER_TOKEN"
pixi run python -m tensorizer_physics.serve --host 127.0.0.1 --port 8765
```

3. Open the SSH tunnel from D1 step 2, and add `http://localhost:8765` with the token in the website.

With the token set and shared mode off, your own code (edited problems, the Workspace, notebooks) runs too. On a
multi-user machine the token matters even on `127.0.0.1`: other users of the same machine can reach that port.

**Be a good neighbour.** Login nodes are shared: run heavy work on a compute node (for example inside an
interactive job, then tunnel to that node) or ask your admins first. Never open the port to the internet, and
follow your institution's rules for running services.

## E. Windows with WSL2

WSL2 is the Linux system built into Windows 10 and 11. The runner runs inside it; Windows browsers reach WSL2's
`localhost` directly. NVIDIA GPUs work in WSL2 with the normal Windows NVIDIA driver.

```powershell
# 1. In PowerShell as Administrator: install WSL2 with Ubuntu (restart Windows if asked)
wsl --install -d Ubuntu
```

```sh
# 2. In the Ubuntu window: turn on systemd (the installer needs it), then leave Ubuntu
printf '[boot]\nsystemd=true\n' | sudo tee /etc/wsl.conf
exit
```

```powershell
# 3. Back in PowerShell: restart WSL so the setting takes effect, then open Ubuntu again
wsl --shutdown
wsl -d Ubuntu
```

```sh
# 4. In Ubuntu: install the runner (add --gpu for an NVIDIA GPU, after installing the NVIDIA Container Toolkit)
curl -fsSL https://jackyko1991.github.io/tensorizer/install.sh | sudo sh -s -- --no-tls
```

5. In the website: **⚙ Backend → Remote → + Add server**, URL `http://localhost:8765`, and the token.

WSL2 stops when no Ubuntu window is open for a while. Open Ubuntu (or run `wsl -d Ubuntu` in PowerShell) before you
use the runner. (Without Docker at all: `pixi run api` in a clone of this repository runs the runner natively.)

## F. Mac (native, uses the Apple GPU)

Run the runner directly on macOS, **not in Docker**: Docker on a Mac runs a Linux virtual machine, which has no
access to the Apple GPU (Metal/MPS). The GPU needs Apple silicon (M1 or later); Intel Macs run on the CPU.

```sh
# 1. In Terminal: install pixi, then get the code and its environment (a few minutes the first time)
curl -fsSL https://pixi.sh/install.sh | sh          # then open a new Terminal window
git clone https://github.com/jackyko1991/tensorizer && cd tensorizer && pixi install
```

### F1. On the same Mac

```sh
# 2. Start the runner for this computer only. No token: the website's Local mode cannot send one.
pixi run python -m tensorizer_physics.serve --port 8765
```

3. In the website: **⚙ Backend → Local**, port `8765`, and **Device: MPS**. *Auto* only picks an NVIDIA GPU, so on
   a Mac it means the CPU; the start-up banner lists `mps (Apple GPU)` when PyTorch can use it.

**float64.** MPS has no float64. A problem whose code uses `float64` runs on the CPU instead (the console says
`device: cpu (this problem uses float64, which MPS does not support)`), and single operations MPS lacks fall back to
the CPU (`PYTORCH_ENABLE_MPS_FALLBACK=1`), so every problem still runs.

### F2. From your other devices (Tailscale)

```sh
# 2. Make a token (copy it somewhere safe), then start the runner with it
export TENSORIZER_TOKEN=$(openssl rand -hex 32); echo "$TENSORIZER_TOKEN"
pixi run python -m tensorizer_physics.serve --host 127.0.0.1 --port 8765

# 3. Install the Tailscale app for macOS (https://tailscale.com/download or the Mac App Store) and log in; in the
#    admin console (https://login.tailscale.com/admin/dns) turn on MagicDNS and "HTTPS Certificates". Then, in a
#    second Terminal window (if `tailscale` is not found, turn on the command-line tool in the app's settings):
tailscale serve --bg 8765
tailscale serve status        # prints https://<mac>.<tailnet>.ts.net
```

4. On each other device: install Tailscale with the same account, then **⚙ Backend → Remote → + Add server** with
   `https://<mac>.<tailnet>.ts.net` and the token.

With a token set, the Local mode of F1 no longer works (every request needs the token); use the remote server
entry on the Mac too. Remote runs use the server's default device, which on a Mac is the CPU: the Device choice
(MPS) applies to Local only.

## G. ARM64 device (Raspberry Pi 4/5, cloud ARM servers)

> The ARM64 image is published with the next runner image build (the `Runner container image` workflow); until
> then `docker pull` on ARM finds no matching image.

Needs a **64-bit OS** (Raspberry Pi OS 64-bit, or Ubuntu 24.04 arm64; `uname -m` prints `aarch64`) and at least
4 GB memory (8 GB is better). Everything runs on the CPU, so the heavy CFD and training problems are slow.

```sh
# 1. The same installer: it installs Docker if needed and pulls the ARM64 image by itself.
#    --gpu is not available on ARM (the CUDA image is x86_64 only).
curl -fsSL https://jackyko1991.github.io/tensorizer/install.sh | sudo sh -s -- --no-tls
```

2. Make it reachable exactly as for a home server: [C1 (Tailscale)](#c1-home-server-with-tailscale-private) or
   [C2 (Cloudflare Tunnel)](#c2-home-server-with-a-cloudflare-tunnel-public-url), then add the `https://` address
   and the token in the website.

**Without Docker** (pixi has a `linux-aarch64` environment):

```sh
curl -fsSL https://pixi.sh/install.sh | sh          # then open a new shell
git clone https://github.com/jackyko1991/tensorizer && cd tensorizer && pixi install
export TENSORIZER_TOKEN=$(openssl rand -hex 32); echo "$TENSORIZER_TOKEN"
pixi run python -m tensorizer_physics.serve --host 127.0.0.1 --port 8765   # keep it running, e.g. in tmux
```

Then `tailscale serve --bg 8765` or a Cloudflare Tunnel as in C1/C2, and add the `https://` address with the token.

---

## Security

- **Pairing token.** The installer makes a random 64-character token, prints it once, and stores only its SHA-256
  (`TENSORIZER_TOKEN_SHA256` in the root-only `/etc/tensorizer/runner.env`). Every request except `GET /api/health`
  must carry `Authorization: Bearer <token>`; the runner compares hashes in constant time, never logs the token,
  and after 10 wrong tokens in a minute from one address answers 429 for a minute. Lost it, or shared it by
  mistake? Run the installer again with `--new-token`. Running the runner yourself: set `TENSORIZER_TOKEN`
  (or `TENSORIZER_TOKEN_SHA256`) in its environment.
- **Shared or private.** By default the runner is set up as *your own* server: generous limits (time, output,
  cache). With `--shared` it uses the public-demo defences: 120 s per run, one job per visitor, a rate limit, and
  no edited or user-written code. Use `--shared` if you hand the URL to a class or post it publicly.
- **Your own code and the AI agent.** On a token-protected private server, edited code, Workspace problems,
  notebooks and the agent's runs are on (the installer writes `ALLOW_CODE_EXEC=1` and `AGENT_ENABLED=1`; set them
  to 0 to turn them off). With `--shared` they are off. A runner **without** a token that listens on a network
  address never runs user code, whatever `ALLOW_CODE_EXEC` says. `/api/health` reports `auth`, `shared` and
  `code_exec`, and the site's Test connection shows them.
- **Firewall.** Only SSH (22) and, for options A/B, HTTP/HTTPS (80, 443) need to be open. Never open 8765: the
  runner listens on `127.0.0.1` only, and Caddy, Tailscale, Cloudflare or SSH carry the traffic to it. With `ufw`:

  ```sh
  sudo ufw allow OpenSSH && sudo ufw allow 80/tcp && sudo ufw allow 443/tcp && sudo ufw enable
  ```

  (`firewalld` users: `sudo firewall-cmd --permanent --add-service=http --add-service=https && sudo firewall-cmd --reload`.)
- **Isolation.** The container runs as an unprivileged user with a read-only system, no Linux capabilities, and all
  scratch files in memory (`/work`), which are gone after a restart. Your problem runs still execute Python code, so
  run the server for people you trust, or with `--shared`.

## Troubleshooting

| Symptom | Check |
|---|---|
| The installer stops at "Waiting for the runner" | `journalctl -u tensorizer-runner -n 50` shows the runner's log |
| `https://…` does not load (A/B) | `journalctl -u tensorizer-caddy -n 50`; ports 80/443 open in the cloud firewall; DNS points at the server |
| The website says "cannot reach" | open `<URL>/api/health` in the browser: it should show `{"status":"ok", …}` |
| GPU not listed | `nvidia-smi` works on the host; NVIDIA Container Toolkit installed; installed with `--gpu` |
| Status of the services | `systemctl status tensorizer-runner tensorizer-caddy` |

Remove everything: `curl -fsSL https://jackyko1991.github.io/tensorizer/install.sh | sudo sh -s -- --uninstall`.
