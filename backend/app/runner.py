"""Alias of tensorizer_physics.runner (the same module object), so the old import path keeps working."""

import sys

from tensorizer_physics import runner as _module

sys.modules[__name__] = _module
