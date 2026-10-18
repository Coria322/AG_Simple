"""EventBus síncrono (patrón Observer, SUP-003).

Se usa en el hilo de la GUI: los controladores drenan la ``queue.Queue`` del hilo de
trabajo y publican cada evento aquí para notificar a las vistas suscritas.
El despacho respeta la herencia: suscribirse a ``DomainEvent`` recibe todos los eventos.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Callable
from typing import Any, TypeVar

E = TypeVar("E")
Handler = Callable[[Any], None]
ErrorHandler = Callable[[BaseException, object, Handler], None]


class EventBus:
    """Bus de eventos con suscripción por tipo.

    Si un manejador falla, el resto igual recibe el evento. Con ``on_error`` el
    fallo se reporta ahí; sin él, al final se lanza un ``ExceptionGroup``.
    """

    def __init__(self, on_error: ErrorHandler | None = None) -> None:
        self._handlers: defaultdict[type, list[Handler]] = defaultdict(list)
        self._lock = threading.Lock()
        self._on_error = on_error

    def subscribe(self, event_type: type[E], handler: Callable[[E], None]) -> Callable[[], None]:
        """Suscribe ``handler`` y devuelve una función para desuscribirlo."""
        with self._lock:
            self._handlers[event_type].append(handler)

        def unsubscribe() -> None:
            with self._lock:
                handlers = self._handlers.get(event_type, [])
                if handler in handlers:
                    handlers.remove(handler)

        return unsubscribe

    def publish(self, event: object) -> int:
        """Entrega ``event`` a los manejadores de su tipo y de sus bases.

        Devuelve cuántos manejadores fueron invocados.
        """
        with self._lock:
            targets = [
                h for cls in type(event).__mro__ for h in tuple(self._handlers.get(cls, ()))
            ]
        errors: list[Exception] = []
        for handler in targets:
            try:
                handler(event)
            except Exception as exc:
                if self._on_error is not None:
                    self._on_error(exc, event, handler)
                else:
                    errors.append(exc)
        if errors:
            raise ExceptionGroup(f"Fallaron {len(errors)} manejadores de {type(event).__name__}",
                                 errors)
        return len(targets)

    def clear(self) -> None:
        with self._lock:
            self._handlers.clear()

    def subscriber_count(self, event_type: type) -> int:
        with self._lock:
            return len(self._handlers.get(event_type, ()))
