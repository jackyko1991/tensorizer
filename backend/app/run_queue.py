"""Alias of tensorizer_physics.run_queue (the same module object), so the old import path keeps working."""

import sys

from tensorizer_physics import run_queue as _module

sys.modules[__name__] = _module
