"""Alias of tensorizer_physics.core.runtime (the same module object), so the old import path keeps working."""

import sys

from tensorizer_physics.core import runtime as _module

sys.modules[__name__] = _module
