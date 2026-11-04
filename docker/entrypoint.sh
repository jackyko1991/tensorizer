#!/bin/bash
# Container entrypoint for the Tensorizer runner image (see the Dockerfile).
#   serve [extra args]  start the runner on $TENSORIZER_HOST:$TENSORIZER_PORT (the default command)
#   anything else       run it inside the pixi environment, e.g. `python -c "import torch"` or `bash`
set -eu

# Everything writable lives under /work (tmpfs or a volume), so the root filesystem can be read-only.
for d in "$TENSORIZER_CACHE" "$TENSORIZER_WORKSPACES" "$AGENT_WORKSPACES" "$MPLCONFIGDIR" "$HOME" \
    "$XDG_CACHE_HOME" "$TMPDIR"; do
    mkdir -p "$d" 2>/dev/null || echo "warning: cannot create $d (mount /work as a tmpfs or a writable volume)" >&2
done

# Activate the pixi environment that was installed at build time.
# shellcheck disable=SC1091
. /usr/local/share/tensorizer/shell-hook.sh

if [ "$#" -eq 0 ] || [ "$1" = "serve" ]; then
    [ "$#" -gt 0 ] && shift
    echo "tensorizer-runner: env=${TENSORIZER_PIXI_ENV} shared=${TENSORIZER_SHARED} port=${TENSORIZER_PORT}" \
        "token=$([ -n "${TENSORIZER_TOKEN:-}" ] && echo set || echo none)"
    exec python -m tensorizer_physics.serve --host "$TENSORIZER_HOST" --port "$TENSORIZER_PORT" "$@"
fi
exec "$@"
