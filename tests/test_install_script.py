"""The server installer (scripts/install.sh, also served from the site as /install.sh) and the container image files.

Nothing here touches the machine: the installer only runs with --dry-run, which prints commands instead of running them.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "install.sh"
PUBLIC = ROOT / "frontend" / "public" / "install.sh"
SH = shutil.which("sh")
needs_sh = pytest.mark.skipif(SH is None, reason="no POSIX sh on this machine")


def dry_run(*args: str) -> str:
    r = subprocess.run(
        [SH, str(SCRIPT), "--dry-run", *args], capture_output=True, text=True, encoding="utf-8", timeout=60
    )
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_public_copy_is_identical():
    # the site serves frontend/public/install.sh; it must be exactly the reviewed script
    assert PUBLIC.read_bytes() == SCRIPT.read_bytes(), "run: cp scripts/install.sh frontend/public/install.sh"


@needs_sh
def test_syntax():
    subprocess.run([SH, "-n", str(SCRIPT)], check=True)


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_shellcheck():
    subprocess.run(["shellcheck", "-s", "sh", str(SCRIPT)], check=True)
    subprocess.run(["shellcheck", str(ROOT / "docker" / "entrypoint.sh")], check=True)


@needs_sh
def test_dry_run_domain_sets_up_caddy_and_service():
    out = dry_run("--domain", "runner.example.org")
    assert " pull ghcr.io/jackyko1991/tensorizer-runner:latest" in out  # docker, or podman where that is installed
    assert "runner.example.org {" in out and "reverse_proxy 127.0.0.1:8765" in out
    assert "/etc/systemd/system/tensorizer-runner.service" in out
    assert "--read-only" in out and "-p 127.0.0.1:8765:8765" in out  # never published on a public interface
    assert "--env-file /etc/tensorizer/runner.env" in out  # the token travels in a root-only file, not argv
    assert "TENSORIZER_SHARED=0" in out and "TRUSTED_PROXY_HOPS=1" in out
    assert "ALLOW_CODE_EXEC=1" in out and "AGENT_ENABLED=1" in out  # own, token-protected server
    assert "systemctl restart tensorizer-caddy.service" in out
    assert "URL   : https://runner.example.org" in out


@needs_sh
def test_dry_run_never_shows_a_real_token():
    out = dry_run("--domain", "runner.example.org")
    assert "TENSORIZER_TOKEN_SHA256=<sha256 of the token>" in out  # the server keeps only the hash
    assert "TENSORIZER_TOKEN=" not in out
    assert "token : <random token" in out  # shown once, on screen
    assert not any(len(w) == 64 and all(c in "0123456789abcdef" for c in w) for w in out.split())


@needs_sh
def test_dry_run_sslip_gpu_shared_port():
    out = dry_run("--sslip", "--ip", "198.51.100.4", "--gpu", "--shared", "--port", "9000", "--version", "0.2.0")
    assert "198-51-100-4.sslip.io {" in out
    assert "tensorizer-runner:0.2.0-cuda" in out
    assert "--gpus all" in out or "nvidia.com/gpu=all" in out
    assert "TENSORIZER_SHARED=1" in out and "TENSORIZER_PORT=9000" in out
    assert "ALLOW_CODE_EXEC=0" in out and "AGENT_ENABLED=0" in out  # shared: no user code
    assert "-p 127.0.0.1:9000:9000" in out


@needs_sh
def test_dry_run_default_is_local_only():
    out = dry_run()
    assert "Caddy" not in out.split("==> Starting")[0].split("Creating the")[1]
    assert "tailscale serve --bg 8765" in out
    assert "TRUSTED_PROXY_HOPS=0" in out


@needs_sh
def test_dry_run_uninstall():
    out = dry_run("--uninstall")
    assert "systemctl disable --now tensorizer-runner.service" in out
    assert "rm -rf /etc/tensorizer /var/lib/tensorizer" in out


@needs_sh
@pytest.mark.parametrize("args", [["--domain", "a.org", "--sslip"], ["--port", "http"], ["--bogus"], ["--domain"]])
def test_bad_arguments_fail(args):
    r = subprocess.run([SH, str(SCRIPT), "--dry-run", *args], capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "error:" in r.stderr


def test_dockerfile_contract():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "USER 1000" in text  # non-root (Hugging Face runs as 1000)
    assert "TENSORIZER_PORT=7860" in text  # the Space's app_port
    assert "TENSORIZER_SHARED=1" in text  # public defaults unless a person's own server turns them off
    assert "ARG PIXI_ENV=default" in text  # --build-arg PIXI_ENV=cuda for the -cuda image
    space = (ROOT / "space" / "README.md").read_text(encoding="utf-8")
    assert "app_port: 7860" in space
    entry = (ROOT / "docker" / "entrypoint.sh").read_bytes()
    assert b"\r\n" not in entry, "entrypoint.sh must have LF line endings"
