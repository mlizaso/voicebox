"""Compatibility alias for :mod:`backend.speak_events`.

The event bus was historically housed under the MCP package even though REST
routes and generation services also use it. Keep the old import path as the
same module object so existing imports and monkeypatches continue to affect the
canonical event bus.
"""

import sys

from .. import speak_events as _speak_events

sys.modules[__name__] = _speak_events
