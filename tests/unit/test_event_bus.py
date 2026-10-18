"""TASK-003 — EventBus síncrono."""

from __future__ import annotations

import pytest

from plataforma_ag.core.event_bus import EventBus
from plataforma_ag.domain.enums import LogLevel
from plataforma_ag.domain.events import DomainEvent, LogRecorded, ProgressChanged


def _progress(n: int = 1) -> ProgressChanged:
    return ProgressChanged(current=n, maximum=10, percent=n * 10.0)


def test_publica_a_suscriptores_del_tipo() -> None:
    bus, got = EventBus(), []
    bus.subscribe(ProgressChanged, got.append)
    assert bus.publish(_progress(3)) == 1
    assert got[0].current == 3


def test_no_entrega_a_otros_tipos() -> None:
    bus, got = EventBus(), []
    bus.subscribe(LogRecorded, got.append)
    assert bus.publish(_progress()) == 0
    assert got == []


def test_suscripcion_a_la_base_recibe_todo() -> None:
    bus, got = EventBus(), []
    bus.subscribe(DomainEvent, got.append)
    bus.publish(_progress())
    bus.publish(LogRecorded(level=LogLevel.INFO, message="hola"))
    assert len(got) == 2


def test_desuscripcion() -> None:
    bus, got = EventBus(), []
    unsubscribe = bus.subscribe(ProgressChanged, got.append)
    unsubscribe()
    unsubscribe()  # idempotente
    bus.publish(_progress())
    assert got == [] and bus.subscriber_count(ProgressChanged) == 0


def test_desuscribirse_durante_la_publicacion_es_seguro() -> None:
    bus, got = EventBus(), []
    unsubscribe = None

    def once(event: ProgressChanged) -> None:
        got.append(event)
        assert unsubscribe is not None
        unsubscribe()

    unsubscribe = bus.subscribe(ProgressChanged, once)
    bus.publish(_progress())
    bus.publish(_progress())
    assert len(got) == 1


def test_un_manejador_fallido_no_impide_a_los_demas() -> None:
    bus, got = EventBus(), []

    def broken(_: object) -> None:
        raise RuntimeError("falla")

    bus.subscribe(ProgressChanged, broken)
    bus.subscribe(ProgressChanged, got.append)
    with pytest.raises(ExceptionGroup) as info:
        bus.publish(_progress())
    assert len(got) == 1
    assert isinstance(info.value.exceptions[0], RuntimeError)


def test_on_error_reporta_sin_lanzar() -> None:
    reported: list[BaseException] = []
    bus = EventBus(on_error=lambda exc, event, handler: reported.append(exc))
    bus.subscribe(ProgressChanged, lambda e: 1 / 0)
    assert bus.publish(_progress()) == 1
    assert isinstance(reported[0], ZeroDivisionError)


def test_clear() -> None:
    bus = EventBus()
    bus.subscribe(ProgressChanged, lambda e: None)
    bus.clear()
    assert bus.publish(_progress()) == 0
