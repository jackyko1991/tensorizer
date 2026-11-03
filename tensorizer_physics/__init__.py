"""Tensorizer's Python core: shared problem helpers, the runner and its limits.

    import tensorizer_physics as tz

The problem library (problems/<category>/<slug>/) stays plain, editable files; this package is the plumbing they
use (tensorizer_physics.core) and the runner that renders and tests them (tensorizer_physics.runner). The old paths
problems._common.* and backend.app.runner / run_queue / limits are aliases of these modules.
"""

__version__ = "0.1.0"
FORMAT_VERSION = 1  # the meta.yaml format this runner understands
