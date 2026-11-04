# Tensorizer runner image: ONE image for the Hugging Face Space, a VPS and a home server (docs/design/backend.md §7).
#
#   CPU  : docker build -t tensorizer-runner .
#   CUDA : docker build --build-arg PIXI_ENV=cuda -t tensorizer-runner:cuda .
#
# CI (.github/workflows/runner-image.yml) publishes ghcr.io/jackyko1991/tensorizer-runner:<version> and
# :<version>-cuda. The Hugging Face Space builds this file as is (CPU, the last stage), see docs/deploy-huggingface.md.
#
# Runtime behaviour, all switchable by environment variables (docker run -e NAME=value, or Space variables):
#   TENSORIZER_PORT    7860 (what Hugging Face expects); the server installer sets 8765
#   TENSORIZER_SHARED  1 = public/shared server defences (the safe default); 0 on a person's own server
#   TENSORIZER_TOKEN   pairing token (read by the runner from step 3 of the design; never passed to user code)
#
# The root filesystem can be read-only: everything the runner writes goes to /work (a tmpfs or a volume), e.g.
#   docker run --read-only --tmpfs /work:uid=1000,gid=1000 --tmpfs /tmp -p 127.0.0.1:8765:8765 \
#     -e TENSORIZER_PORT=8765 -e TENSORIZER_SHARED=0 tensorizer-runner
# No BuildKit-only syntax here, so the Hugging Face builder handles it too.

ARG PIXI_ENV=default

# ---- build: solve and install the pixi environment ---------------------------------------------------------------
FROM ghcr.io/prefix-dev/pixi:0.70.0-bookworm AS build
ARG PIXI_ENV
WORKDIR /app
# The CUDA environment declares system-requirements cuda = "12"; the build machine has no GPU, so say it has one.
ENV CONDA_OVERRIDE_CUDA=12.0
COPY pixi.toml pixi.lock ./
RUN pixi install --locked -e "${PIXI_ENV}" && rm -rf /root/.cache
# Activation script, so the runtime image needs no pixi (and never writes to .pixi at start-up).
RUN pixi shell-hook --locked -e "${PIXI_ENV}" -s bash > /shell-hook.sh
COPY . .
# Byte-compile now: with a read-only root filesystem Python cannot write __pycache__ later.
RUN . /shell-hook.sh && python -m compileall -q -j 0 tensorizer_physics backend problems >/dev/null || true

# ---- runtime -----------------------------------------------------------------------------------------------------
FROM debian:bookworm-slim AS runtime
ARG PIXI_ENV
# Hugging Face runs the container as user 1000; so does the server installer.
RUN useradd -m -u 1000 user \
    && mkdir -p /work \
    && chown user:user /work
COPY --from=build /app /app
COPY --from=build /shell-hook.sh /usr/local/share/tensorizer/shell-hook.sh
COPY docker/entrypoint.sh /usr/local/bin/tensorizer-entrypoint
RUN chmod 0755 /usr/local/bin/tensorizer-entrypoint
WORKDIR /app
# /app stays owned by root: the running user can read the library but not change it.
USER 1000

ENV TENSORIZER_PIXI_ENV=${PIXI_ENV} \
    TENSORIZER_PORT=7860 \
    TENSORIZER_HOST=0.0.0.0 \
    TENSORIZER_SHARED=1 \
    CORS_ORIGINS=https://jackyko1991.github.io \
    TENSORIZER_CACHE=/work/runs \
    TENSORIZER_WORKSPACES=/work/ws \
    AGENT_WORKSPACES=/work/agent \
    TENSORIZER_MAX_CONCURRENT=1 \
    TENSORIZER_MAX_QUEUE=20 \
    TENSORIZER_PER_CLIENT=1 \
    AGENT_ENABLED=1 \
    MAX_AGENT_SESSIONS=2 \
    OMP_NUM_THREADS=2 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/work/mpl \
    HOME=/work/home \
    XDG_CACHE_HOME=/work/cache \
    TMPDIR=/work/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 7860 8765
ENTRYPOINT ["/usr/local/bin/tensorizer-entrypoint"]
CMD ["serve"]
