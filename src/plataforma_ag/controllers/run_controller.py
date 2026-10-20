"""Controlador de ejecución para la capa de presentación (RF-17, RNF-01).

Conecta la máquina de estados y el WorkerThread del núcleo con la interfaz gráfica,
despachando eventos periódicamente al hilo principal de Tkinter.
"""

from __future__ import annotations

import contextlib
import queue
from collections.abc import Callable

from plataforma_ag.controllers.telemetry_controller import TelemetryController
from plataforma_ag.core.engine.run_controller import RunController as CoreRunController
from plataforma_ag.domain.config import ExperimentConfig
from plataforma_ag.domain.enums import RunState
from plataforma_ag.domain.events import DomainEvent, RunStateChanged


class RunController:
    """Controlador que vincula la ejecución del AG con las vistas de la GUI."""

    def __init__(self, telemetry_controller: TelemetryController) -> None:
        self.telemetry = telemetry_controller
        self._core_controller = CoreRunController()
        self._state_change_listeners: list[Callable[[RunState], None]] = []

    @property
    def state(self) -> RunState:
        return self._core_controller.state

    def add_state_listener(self, cb: Callable[[RunState], None]) -> None:
        if cb not in self._state_change_listeners:
            self._state_change_listeners.append(cb)

    def _notify_state(self, state: RunState) -> None:
        for cb in list(self._state_change_listeners):
            with contextlib.suppress(Exception):
                cb(state)

    def start_run(self, config: ExperimentConfig) -> None:
        """Inicia una nueva corrida del algoritmo genético."""
        self._core_controller.start(config)
        self._notify_state(self._core_controller.state)

    def pause_run(self) -> None:
        """Pausa la corrida activa."""
        self._core_controller.pause()
        self._notify_state(self._core_controller.state)

    def resume_run(self) -> None:
        """Reanuda la corrida pausada."""
        self._core_controller.resume()
        self._notify_state(self._core_controller.state)

    def abort_run(self) -> None:
        """Cancela y aborta la corrida."""
        self._core_controller.abort()
        self._notify_state(self._core_controller.state)

    def poll_events(self) -> int:
        """Drena y procesa todos los eventos pendientes en la cola del WorkerThread.

        Devuelve la cantidad de eventos procesados en este ciclo.
        """
        count = 0
        q = self._core_controller.event_queue
        while True:
            try:
                event: DomainEvent = q.get_nowait()
                count += 1
                self.telemetry.handle_event(event)

                if isinstance(event, RunStateChanged):
                    self._notify_state(event.current)
            except queue.Empty:
                break

        return count
