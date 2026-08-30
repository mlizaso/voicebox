"""In-memory pub/sub for speaking-pill SSE broadcasts.

MCP ``voicebox.speak`` calls and the REST ``POST /speak`` route publish
start/end events that DictateWindow subscribes to via ``/events/speak``, so the
floating pill surfaces whenever an agent is speaking.

This module intentionally lives outside both the MCP and HTTP route packages:
the event bus is shared application infrastructure, not an MCP implementation
detail.
"""

import asyncio
from contextlib import suppress
from typing import Any

# Each subscriber gets its own queue. New events are dropped if a client lags.
_subscribers: set[asyncio.Queue[dict[str, Any]]] = set()


def subscribe() -> asyncio.Queue[dict[str, Any]]:
    """Register a new subscriber; caller must call unsubscribe() when done."""
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=64)
    _subscribers.add(queue)
    return queue


def unsubscribe(queue: asyncio.Queue[dict[str, Any]]) -> None:
    _subscribers.discard(queue)


def publish(kind: str, payload: dict[str, Any]) -> None:
    """Fan out to all current subscribers without blocking.

    Each subscriber gets its own dict copy because consumers are allowed to
    mutate their event while serializing it. A full subscriber queue drops the
    new event rather than delaying publishers or other subscribers.
    """
    for queue in list(_subscribers):
        event = {"kind": kind, **payload}
        # Slow subscribers must not block publishers or other subscribers.
        with suppress(asyncio.QueueFull):
            queue.put_nowait(event)
