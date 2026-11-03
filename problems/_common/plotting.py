"""Alias of tensorizer_physics.core.plotting (the same module object), so the old import path keeps working."""

import sys

from tensorizer_physics.core import plotting as _module

sys.modules[__name__] = _module
