"""Focused contracts for the shared speak-event bus."""

import importlib

from backend import build_binary, speak_events


def test_publish_isolated_event_per_subscriber() -> None:
    first = speak_events.subscribe()
    second = speak_events.subscribe()
    payload = {"generation_id": "gen-1"}
    try:
        speak_events.publish("speak-start", payload)

        first_event = first.get_nowait()
        second_event = second.get_nowait()
        first_event.pop("kind")

        assert second_event == {
            "kind": "speak-start",
            "generation_id": "gen-1",
        }
        assert payload == {"generation_id": "gen-1"}
    finally:
        speak_events.unsubscribe(first)
        speak_events.unsubscribe(second)


def test_unsubscribe_stops_delivery() -> None:
    queue = speak_events.subscribe()
    speak_events.unsubscribe(queue)

    speak_events.publish("speak-end", {"generation_id": "gen-1"})

    assert queue.empty()


def test_legacy_mcp_import_is_canonical_module(monkeypatch) -> None:
    legacy_events = importlib.import_module("backend.mcp_server.events")

    assert legacy_events is speak_events
    assert importlib.import_module("backend.mcp_server.tools").mcp_events is speak_events
    assert importlib.import_module("backend.routes.events").speak_event_bus is speak_events
    assert importlib.import_module("backend.routes.speak").speak_event_bus is speak_events

    def replacement(*_args, **_kwargs) -> None:
        pass

    monkeypatch.setattr(legacy_events, "publish", replacement)
    assert speak_events.publish is replacement


def test_frozen_server_collects_canonical_and_legacy_event_modules(monkeypatch) -> None:
    captured_args: list[list[str]] = []

    def capture_pyinstaller_args(args: list[str]) -> None:
        captured_args.append(args)

    monkeypatch.setattr(build_binary.PyInstaller.__main__, "run", capture_pyinstaller_args)
    monkeypatch.setattr(build_binary.os, "chdir", lambda _path: None)
    monkeypatch.setattr(build_binary.platform, "system", lambda: "Linux")
    monkeypatch.setattr(build_binary, "is_apple_silicon", lambda: False)

    build_binary.build_server()

    args = captured_args[0]
    hidden_imports = {args[index + 1] for index, argument in enumerate(args[:-1]) if argument == "--hidden-import"}
    assert {
        "backend.mcp_server.events",
        "backend.speak_events",
    } <= hidden_imports


def test_generation_completion_uses_shared_event_bus(monkeypatch) -> None:
    generation_service = importlib.import_module("backend.services.generation")
    published: list[tuple[str, dict[str, str]]] = []

    def capture_publish(kind: str, payload: dict[str, str]) -> None:
        published.append((kind, payload))

    monkeypatch.setattr(speak_events, "publish", capture_publish)

    generation_service._notify_speak_end("gen-1", status="completed")

    assert published == [
        (
            "speak-end",
            {"generation_id": "gen-1", "status": "completed"},
        )
    ]
