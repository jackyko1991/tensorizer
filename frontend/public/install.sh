#!/bin/sh
# Tensorizer runner: server installer (docs/remote-server.md walks through every case).
#
#   curl -fsSL https://jackyko1991.github.io/tensorizer/install.sh | sudo sh -s -- [options]
#
# What it does, in order (re-running it is safe; it updates the image and keeps your token):
#   1. installs Docker with Docker's official script, unless Docker or Podman is already there
#   2. pulls the runner image  ghcr.io/jackyko1991/tensorizer-runner:<version>[-cuda]
#   3. writes /etc/tensorizer/runner.env (port, shared mode, pairing token; readable by root only)
#   4. creates and starts the systemd service  tensorizer-runner  (the container, listening on 127.0.0.1 only)
#   5. optionally runs Caddy (also in a container, service tensorizer-caddy) for automatic HTTPS
#   6. prints the URL to paste in the website (Backend > Remote > Add server) and, the first time, the pairing token
#
# Options:
#   --domain NAME     HTTPS for NAME with Caddy (point NAME's DNS A record at this server first)
#   --sslip           HTTPS without a domain: Caddy on <public-ip>.sslip.io (use --ip ADDR to choose the address)
#   --no-tls          no Caddy; the runner stays on 127.0.0.1 for Tailscale, a Cloudflare Tunnel or an SSH tunnel
#                     (the default when no mode is given)
#   --ip ADDR         the public address for --sslip (default: detected)
#   --port N          local port of the runner (default 8765)
#   --gpu             NVIDIA GPU: the -cuda image (needs the NVIDIA driver and the NVIDIA Container Toolkit;
#                     x86_64 only: on ARM64 the installer stops, the CPU image runs there instead)
#   --shared          public/shared-server defences (TENSORIZER_SHARED=1); default off: your own server
#   --version TAG     image version (default latest)
#   --image REF       a different image (default ghcr.io/jackyko1991/tensorizer-runner)
#   --memory SIZE     memory limit for the runner container, e.g. 4g (default: none)
#   --cpus N          CPU limit for the runner container, e.g. 2 (default: none)
#   --new-token       replace the pairing token (the old one stops working)
#   --uninstall       remove the services, containers, image, Caddy certificates and /etc/tensorizer
#   --dry-run         print every command and file instead of running or writing it (no root needed)
#   -h, --help        this help
#
# The CPU image is multi-arch (x86_64 and ARM64: Raspberry Pi 4/5 with a 64-bit OS, Graviton, Ampere); the CUDA
# image is x86_64 only.
set -eu

# ---- defaults ------------------------------------------------------------------------------------------------------
IMAGE="ghcr.io/jackyko1991/tensorizer-runner"
VERSION="latest"
PORT=8765
MODE=""            # domain | sslip | notls
DOMAIN=""
IP=""
GPU=0
SHARED=0
MEMORY=""
CPUS=""
NEW_TOKEN=0
UNINSTALL=0
DRY_RUN=0
CONF_DIR=/etc/tensorizer
ENV_FILE=$CONF_DIR/runner.env
CADDYFILE=$CONF_DIR/Caddyfile
DATA_DIR=/var/lib/tensorizer
UNIT_DIR=/etc/systemd/system
RUNNER=tensorizer-runner
CADDY=tensorizer-caddy
CADDY_IMAGE="docker.io/library/caddy:2"

say() { printf '%s\n' "$*"; }
step() { printf '\n==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# The help is the comment block at the top of this file (piped through `sh -s` there is no file to read).
usage() {
    if [ -r "$0" ]; then
        sed -n '2,/^set -eu/p' "$0" | sed -e '/^set -eu/d' -e 's/^# \{0,1\}//'
    else
        say "Options: see the top of https://jackyko1991.github.io/tensorizer/install.sh"
    fi
}

# ---- arguments -----------------------------------------------------------------------------------------------------
need_arg() { [ "$#" -ge 2 ] && [ -n "$2" ] || die "$1 needs a value (see --help)"; }
set_mode() { [ -z "$MODE" ] || [ "$MODE" = "$1" ] || die "choose one of --domain, --sslip, --no-tls"; MODE=$1; }
while [ "$#" -gt 0 ]; do
    case "$1" in
        --domain) need_arg "$@"; set_mode domain; DOMAIN=$2; shift ;;
        --sslip) set_mode sslip ;;
        --no-tls) set_mode notls ;;
        --ip) need_arg "$@"; IP=$2; shift ;;
        --port) need_arg "$@"; PORT=$2; shift ;;
        --gpu) GPU=1 ;;
        --shared) SHARED=1 ;;
        --version) need_arg "$@"; VERSION=$2; shift ;;
        --image) need_arg "$@"; IMAGE=$2; shift ;;
        --memory) need_arg "$@"; MEMORY=$2; shift ;;
        --cpus) need_arg "$@"; CPUS=$2; shift ;;
        --new-token) NEW_TOKEN=1 ;;
        --uninstall) UNINSTALL=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h | --help) usage; exit 0 ;;
        *) die "unknown option: $1 (see --help)" ;;
    esac
    shift
done
[ -n "$MODE" ] || MODE=notls
case "$PORT" in '' | *[!0-9]*) die "--port must be a number" ;; esac
case "$DOMAIN" in *[!A-Za-z0-9.-]*) die "--domain: letters, digits, dots and dashes only" ;; esac

# CPU architecture (TENSORIZER_INSTALL_ARCH overrides uname -m, for tests). The CPU image is multi-arch, so
# `docker pull` picks the right one by itself; only the CUDA image is x86_64 only.
ARCH=${TENSORIZER_INSTALL_ARCH:-$(uname -m)}
if [ "$GPU" -eq 1 ]; then
    case "$ARCH" in
        aarch64 | arm64 | armv8* | armv7* | armv6*)
            die "--gpu: the CUDA image is x86_64 (amd64) only and this machine is $ARCH. Run without --gpu: the CPU image works on ARM64." ;;
    esac
fi

TAG=$VERSION
[ "$GPU" -eq 1 ] && TAG="$VERSION-cuda"
[ "$GPU" -eq 1 ] && [ "$VERSION" = latest ] && TAG=cuda
REF="$IMAGE:$TAG"

# ---- running commands (or printing them with --dry-run) -----------------------------------------------------------
SUDO=""
if [ "$DRY_RUN" -eq 0 ] && [ "$(id -u)" -ne 0 ]; then
    command -v sudo >/dev/null 2>&1 || die "run as root (or install sudo)"
    SUDO=sudo
fi

# run CMD...: run as root, or print it with --dry-run
run() {
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '+ %s\n' "$*"
    else
        $SUDO "$@"
    fi
}

# write_file PATH MODE < content: write a root-owned file, or print it with --dry-run
write_file() {
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '+ write %s (mode %s):\n' "$1" "$2"
        sed 's/^/    | /'
    else
        $SUDO mkdir -p "$(dirname "$1")"
        # shellcheck disable=SC2016  # $1 and $2 expand in the inner shell, on purpose
        $SUDO sh -c 'umask 077; cat > "$1"; chmod "$2" "$1"' sh "$1" "$2"
    fi
}

have() { command -v "$1" >/dev/null 2>&1; }

# ---- container runtime: Docker, or Podman if that is what this machine has --------------------------------------
pick_runtime() {
    if have docker; then CT=docker
    elif have podman; then CT=podman
    else CT=""
    fi
}

install_docker() {
    step "Installing Docker (Docker's official script, https://get.docker.com)"
    have curl || die "curl is needed to install Docker"
    if [ "$DRY_RUN" -eq 1 ]; then
        say "+ curl -fsSL https://get.docker.com | sh"
    else
        curl -fsSL https://get.docker.com | $SUDO sh
    fi
    run systemctl enable --now docker
    CT=docker
}

# ---- uninstall -----------------------------------------------------------------------------------------------------
uninstall() {
    pick_runtime
    step "Removing the Tensorizer runner"
    for svc in "$CADDY" "$RUNNER"; do
        run systemctl disable --now "$svc.service" || true
        run rm -f "$UNIT_DIR/$svc.service"
    done
    run systemctl daemon-reload
    if [ -n "$CT" ]; then
        run "$CT" rm -f "$RUNNER" "$CADDY" || true
        # every tag of the runner image (CPU and -cuda), not just the one named on this command line
        for ref in $("$CT" images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep "^$IMAGE:" || true); do
            run "$CT" rmi "$ref" || true
        done
    fi
    run rm -rf "$CONF_DIR" "$DATA_DIR"
    say ""
    say "Removed. Docker/Podman itself is left installed. Firewall rules you opened for 80/443 are left as they are."
}

if [ "$UNINSTALL" -eq 1 ]; then
    uninstall
    exit 0
fi

# ---- checks --------------------------------------------------------------------------------------------------------
[ "$(uname -s)" = Linux ] || [ "$DRY_RUN" -eq 1 ] || die "this installer is for Linux (on Windows use WSL2, see docs/remote-server.md)"
have systemctl || [ "$DRY_RUN" -eq 1 ] || die "systemd is needed (on WSL2 enable it in /etc/wsl.conf, see docs/remote-server.md)"

if [ "$MODE" = sslip ] && [ -z "$IP" ]; then
    if [ "$DRY_RUN" -eq 1 ]; then
        IP=203.0.113.7  # documentation address, so a dry run needs no network
    else
        IP=$(curl -fsS4 --max-time 10 https://api.ipify.org || true)
        [ -n "$IP" ] || die "could not detect the public IP address; pass --ip ADDR"
    fi
fi
case "$IP" in *[!0-9.]*) die "--ip: an IPv4 address such as 203.0.113.7" ;; esac
[ "$MODE" = sslip ] && DOMAIN="$(printf '%s' "$IP" | tr . -).sslip.io"

# ---- 1. Docker / Podman --------------------------------------------------------------------------------------------
pick_runtime
if [ -z "$CT" ]; then install_docker; else step "Using $CT ($(command -v "$CT"))"; fi
CT_BIN=$(command -v "$CT" 2>/dev/null || echo "/usr/bin/$CT")

GPU_ARGS=""
if [ "$GPU" -eq 1 ]; then
    if [ "$CT" = podman ]; then GPU_ARGS="--device nvidia.com/gpu=all"; else GPU_ARGS="--gpus all"; fi
    have nvidia-smi || say "warning: nvidia-smi not found; install the NVIDIA driver and the NVIDIA Container Toolkit"
fi

# ---- 2. image ------------------------------------------------------------------------------------------------------
step "Pulling $REF"
run "$CT" pull "$REF"

# ---- 3. settings and pairing token ---------------------------------------------------------------------------------
# PAIRING TOKEN (design step 3). The installer makes a random token, shows it on screen once, and gives the runner
# only its SHA-256 (TENSORIZER_TOKEN_SHA256) through the root-only env file below; the runner compares the hash of
# what the website sends. The token itself is never written anywhere (not in a file, the unit, argv or a log).
new_token() {
    if have openssl; then openssl rand -hex 32
    else od -An -N32 -tx1 /dev/urandom | tr -d ' \n'
    fi
}
# sha256_hex TEXT: the hex SHA-256 of TEXT (printf is a shell builtin, so TEXT is not on any command line)
sha256_hex() {
    if have sha256sum; then printf '%s' "$1" | sha256sum | cut -d' ' -f1
    else printf '%s' "$1" | openssl dgst -sha256 -r | cut -d' ' -f1
    fi
}

TOKEN=""
TOKEN_HASH=""
TOKEN_IS_NEW=0
if [ "$DRY_RUN" -eq 1 ]; then
    TOKEN="<random token from openssl rand -hex 32>"
    TOKEN_HASH="<sha256 of the token>"
    TOKEN_IS_NEW=1
elif [ "$NEW_TOKEN" -eq 0 ] && $SUDO test -f "$ENV_FILE"; then
    TOKEN_HASH=$($SUDO sed -n 's/^TENSORIZER_TOKEN_SHA256=//p' "$ENV_FILE")
    if [ -z "$TOKEN_HASH" ]; then  # an older install kept the token itself: keep it working, store only its hash
        OLD_TOKEN=$($SUDO sed -n 's/^TENSORIZER_TOKEN=//p' "$ENV_FILE")
        [ -n "$OLD_TOKEN" ] && TOKEN_HASH=$(sha256_hex "$OLD_TOKEN")
        OLD_TOKEN=""
    fi
fi
if [ -z "$TOKEN_HASH" ]; then
    TOKEN=$(new_token)
    TOKEN_HASH=$(sha256_hex "$TOKEN")
    TOKEN_IS_NEW=1
fi

PROXY_HOPS=0
[ "$MODE" != notls ] && PROXY_HOPS=1  # Caddy in front: trust exactly one X-Forwarded-For hop
# Your own server (token-protected): edited code, the Workspace, notebooks and the AI agent are on.
# --shared: off (the public-demo defences).
CODE=1
[ "$SHARED" -eq 1 ] && CODE=0

step "Writing $ENV_FILE"
write_file "$ENV_FILE" 600 <<EOF
# Tensorizer runner settings (written by install.sh; re-run the installer to change them)
TENSORIZER_PORT=$PORT
TENSORIZER_HOST=0.0.0.0
TENSORIZER_SHARED=$SHARED
TRUSTED_PROXY_HOPS=$PROXY_HOPS
CORS_ORIGINS=https://jackyko1991.github.io
ALLOW_CODE_EXEC=$CODE
AGENT_ENABLED=$CODE
TENSORIZER_TOKEN_SHA256=$TOKEN_HASH
EOF

# ---- 4. runner service ---------------------------------------------------------------------------------------------
LIMITS=""
[ -n "$MEMORY" ] && LIMITS="$LIMITS --memory $MEMORY"
[ -n "$CPUS" ] && LIMITS="$LIMITS --cpus $CPUS"

step "Creating the $RUNNER service"
write_file "$UNIT_DIR/$RUNNER.service" 644 <<EOF
[Unit]
Description=Tensorizer runner (container $REF)
Wants=network-online.target
After=network-online.target

[Service]
Restart=always
RestartSec=5
ExecStartPre=-$CT_BIN rm -f $RUNNER
ExecStart=$CT_BIN run --rm --name $RUNNER \\
    --read-only --tmpfs /work:rw,uid=1000,gid=1000,mode=0700,size=4g --tmpfs /tmp:rw,size=256m \\
    --cap-drop ALL --security-opt no-new-privileges --pids-limit 1024$LIMITS $GPU_ARGS \\
    --env-file $ENV_FILE -p 127.0.0.1:$PORT:$PORT $REF
ExecStop=$CT_BIN stop -t 10 $RUNNER

[Install]
WantedBy=multi-user.target
EOF

# ---- 5. Caddy for HTTPS (optional) ---------------------------------------------------------------------------------
if [ "$MODE" != notls ]; then
    step "Setting up Caddy for https://$DOMAIN"
    write_file "$CADDYFILE" 644 <<EOF
# Caddy gets and renews the certificate for $DOMAIN automatically (ports 80 and 443 must reach this server).
$DOMAIN {
    reverse_proxy 127.0.0.1:$PORT
}
EOF
    run mkdir -p "$DATA_DIR/caddy"
    write_file "$UNIT_DIR/$CADDY.service" 644 <<EOF
[Unit]
Description=Caddy HTTPS proxy for the Tensorizer runner
Wants=network-online.target
After=network-online.target $RUNNER.service

[Service]
Restart=always
RestartSec=5
ExecStartPre=-$CT_BIN rm -f $CADDY
ExecStart=$CT_BIN run --rm --name $CADDY --network host \\
    -v $CADDYFILE:/etc/caddy/Caddyfile:ro -v $DATA_DIR/caddy:/data $CADDY_IMAGE
ExecStop=$CT_BIN stop -t 10 $CADDY

[Install]
WantedBy=multi-user.target
EOF
    run "$CT" pull "$CADDY_IMAGE"
    if have ufw && $SUDO ufw status 2>/dev/null | grep -q "Status: active"; then
        step "Opening ports 80 and 443 in ufw"
        run ufw allow 80/tcp
        run ufw allow 443/tcp
    fi
else
    # Switching from HTTPS mode back to --no-tls: stop the old Caddy service.
    if [ -f "$UNIT_DIR/$CADDY.service" ]; then
        run systemctl disable --now "$CADDY.service"
        run rm -f "$UNIT_DIR/$CADDY.service" "$CADDYFILE"
    fi
fi

# ---- start ---------------------------------------------------------------------------------------------------------
step "Starting"
run systemctl daemon-reload
run systemctl enable "$RUNNER.service"
run systemctl restart "$RUNNER.service"
if [ "$MODE" != notls ]; then
    run systemctl enable "$CADDY.service"
    run systemctl restart "$CADDY.service"
fi

if [ "$DRY_RUN" -eq 0 ]; then
    printf 'Waiting for the runner'
    i=0
    until curl -fsS "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; do
        i=$((i + 1))
        [ "$i" -ge 60 ] && { say ""; die "no answer after 2 minutes; see: journalctl -u $RUNNER -n 50"; }
        printf '.'
        sleep 2
    done
    say " ready."
fi

# ---- 6. what to paste in the website -------------------------------------------------------------------------------
if [ "$MODE" = notls ]; then URL="http://127.0.0.1:$PORT (local only)"; else URL="https://$DOMAIN"; fi
say ""
say "------------------------------------------------------------------------"
say "  Tensorizer runner is installed."
say "  URL   : $URL"
if [ "$TOKEN_IS_NEW" -eq 1 ]; then
    say "  token : $TOKEN"
    say "          (shown only now; copy it into the website: Backend > Remote > Add server)"
else
    say "  token : unchanged (use --new-token to make a new one)"
fi
say "  shared: $([ "$SHARED" -eq 1 ] && echo 'yes (public defences on)' || echo 'no (your own server)')"
say "  logs  : journalctl -u $RUNNER -f"
say "  remove: re-run this installer with --uninstall"
if [ "$MODE" = notls ]; then
    say ""
    say "  Next, make it reachable over HTTPS (docs/remote-server.md):"
    say "    Tailscale        : tailscale serve --bg $PORT"
    say "    Cloudflare Tunnel: cloudflared tunnel --url http://127.0.0.1:$PORT"
    say "    SSH tunnel       : ssh -N -L $PORT:localhost:$PORT <this server>, then add http://localhost:$PORT"
fi
say "------------------------------------------------------------------------"
