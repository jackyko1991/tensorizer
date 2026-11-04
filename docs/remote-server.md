# Run Tensorizer on your own server

The website (<https://jackyko1991.github.io/tensorizer/>) is a static page. When you press **Run**, it sends the
problem to a *runner*: a small program that runs the Python code and sends back the animation. This page shows how
to put a runner on a machine you control (a rented VPS, a mini-PC at home, a lab server or a Windows PC) and connect
the website to it.

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
| Lab server behind a university/company VPN | an SSH tunnel | `--no-tls` | [D](#d-lab-server-behind-a-vpn-ssh-tunnel) |
| Windows PC | WSL2 | `--no-tls` | [E](#e-windows-with-wsl2) |

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
| `--gpu` | use the NVIDIA GPU (installs the `-cuda` image; needs the NVIDIA driver and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)) |
| `--port 8765` | the runner's local port |
| `--memory 4g`, `--cpus 2` | cap what the runner may use |
| `--shared` | turn on the public-demo defences (time limits, one job per visitor, no edited code); see [Security](#security) |
| `--new-token` | make a new pairing token (the old one stops working) |
| `--version 0.1.0` | a specific runner version instead of `latest` |
| `--uninstall` | remove everything the installer made |

**Updating:** run the same install command again. It downloads the newest image, restarts the service and keeps
your token.

## Connect the website

1. Open <https://jackyko1991.github.io/tensorizer/> → **⚙ Compute** → **Add server**.
2. Paste the URL and the token the installer printed, then **Test connection**. You should see the runner
   version and the devices (`cpu`, or `cpu, cuda` with `--gpu`).
3. Make it the active target. Runs, the Workspace and notebooks now use your server.

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

## D. Lab server behind a VPN (SSH tunnel)

Your computer reaches the lab server over the institute VPN, but the server has no HTTPS name. An **SSH tunnel**
makes the remote runner appear on your own computer as `localhost:8765`, which browsers allow.

```sh
# 1. On the lab server (ask your admin if you have no sudo; Podman is used if it is installed instead of Docker):
curl -fsSL https://jackyko1991.github.io/tensorizer/install.sh | sudo sh -s -- --no-tls          # add --gpu for NVIDIA GPUs

# 2. On your own computer, with the VPN connected; keep this terminal open while you work:
ssh -N -L 8765:localhost:8765 you@lab-server.example.org
#    -L 8765:localhost:8765  forwards your port 8765 to port 8765 on the lab server
#    -N                      no remote shell, just the tunnel
```

3. In the website, use **This device** (`http://localhost:8765`) and paste the token.

If something on your computer already uses port 8765 (for example a local runner), forward another one:
`ssh -N -L 8766:localhost:8765 …` and use `http://localhost:8766`.

Alternatives when SSH is not allowed: Tailscale (C1) if your institute permits it, or ask IT for an internal
HTTPS name and certificate and point a reverse proxy at `127.0.0.1:8765`.

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

5. In the website, use **This device** (`http://localhost:8765`) and paste the token.

WSL2 stops when no Ubuntu window is open for a while. Open Ubuntu (or run `wsl -d Ubuntu` in PowerShell) before you
use the runner. (Without Docker at all: `pixi run api` in a clone of this repository runs the runner natively.)

---

## Security

- **Pairing token.** The installer makes a random 64-character token and gives it to the runner; the website must
  send it with every request except the health check. It is stored only in `/etc/tensorizer/runner.env`
  (root-only) and printed once. Lost it, or shared it by mistake? Run the installer again with `--new-token`.
  *Note: the runner starts checking the token with the pairing step of the backend redesign (step 3 in
  docs/design/backend.md). Until that release, treat a public URL as open to anyone who finds it and prefer the
  private set-ups (Tailscale, SSH tunnel).*
- **Shared or private.** By default the runner is set up as *your own* server: generous limits (time, output,
  cache). With `--shared` it uses the public-demo defences: 120 s per run, one job per visitor, a rate limit, and
  no edited or user-written code. Use `--shared` if you hand the URL to a class or post it publicly.
- **Edited code and the AI agent** are off in the container for now (`ALLOW_CODE_EXEC` and `AGENT_ENABLED` unset);
  library problems with your parameters run. They will be enabled for token-protected private servers once the
  token check is in place.
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
