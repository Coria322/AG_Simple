"""Controlador de ejecución y máquina de estados del motor AG (RF-17, RNF-01).

Implementa la máquina de estados formal con transiciones:
  IDLE -> start() -> RUNNING
  RUNNING -> pause() -> PAUSED
  RUNNING -> abort() -> CANCELLED
  RUNNING -> [evento] -> COMPLETED | ERROR
  PAUSED -> resume() -> RUNNING
  PAUSED -> abort() -> CANCELLED
  COMPLETED | CANCELLED | ERROR -> start() -> RUNNING (reinicio)
"""

from __future__ import annotations

import queue
import threading
from typing import Any

from plataforma_ag.core.engine.ga_engine import GAEngine
from plataforma_ag.core.engine.worker import WorkerThread
from plataforma_ag.domain.config import ExperimentConfig
from plataforma_ag.domain.enums import RunState
from plataforma_ag.domain.errors import InvalidStateTransition
from plataforma_ag.domain.events import (
    DomainEvent,
    RunCancelled,
    RunCompleted,
    RunFailed,
    RunPaused,
    RunResumed,
    RunStarted,
    RunStateChanged,
)


class RunController:
    """Controlador central de la ejecución del AG con máquina de estados finita."""

    def __init__(self) -> None:
        self._state: RunState = RunState.IDLE
        self._worker: WorkerThread | None = None
        self._queue: queue.Queue[DomainEvent] = queue.Queue()
        self._lock = threading.Lock()

    @property
    def state(self) -> RunState:
        """Estado actual de la máquina de estados."""
        with self._lock:
            return self._state

    @property
    def event_queue(self) -> queue.Queue[DomainEvent]:
        """Cola de eventos sincronizada que alimenta la GUI."""
        return self._queue

    def start(self, config: ExperimentConfig, **kwargs: Any) -> None:
        """Inicia una corrida a partir de un ExperimentConfig."""
        with self._lock:
            if self._state in (RunState.RUNNING, RunState.PAUSED):
                raise InvalidStateTransition(self._state, "iniciar")

            # Construir motor y worker
            engine = GAEngine.from_config(config, **kwargs)
            self._worker = WorkerThread(engine=engine, event_queue=self._queue)
            self._transition_to(RunState.RUNNING)
            self._worker.start()

    def pause(self) -> None:
        """Pausa la corrida activa."""
        with self._lock:
            if self._state is not RunState.RUNNING:
                raise InvalidStateTransition(self._state, "pausar")
            if self._worker is not None:
                self._worker.pause()
            self._transition_to(RunState.PAUSED)

    def resume(self) -> None:
        """Reanuda la corrida pausada."""
        with self._lock:
            if self._state is not RunState.PAUSED:
                raise InvalidStateTransition(self._state, "reanudar")
            if self._worker is not None:
                self._worker.resume()
            self._transition_to(RunState.RUNNING)

    def abort(self) -> None:
        """Cancela y detiene la corrida."""
        with self._lock:
            if self._state not in (RunState.RUNNING, RunState.PAUSED):
                raise InvalidStateTransition(self._state, "abortar")
            if self._worker is not None:
                self._worker.cancel()
            self._transition_to(RunState.CANCELLED)

    def wait_until_finished(self, timeout: float | None = 2.0) -> bool:
        """Espera a que el hilo de trabajo termine. Devuelve True si terminó con éxito."""
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=timeout)
            return not self._worker.is_alive()
        return True

    def process_event(self, event: DomainEvent) -> None:
        """Actualiza el estado interno ante eventos emitidos por el hilo de trabajo."""
        with self._lock:
            if self._state in (RunState.COMPLETED, RunState.CANCELLED, RunState.ERROR):
                return
            if isinstance(event, RunStarted):
                self._transition_to(RunState.RUNNING)
            elif isinstance(event, RunPaused):
                self._transition_to(RunState.PAUSED)
            elif isinstance(event, RunResumed):
                self._transition_to(RunState.RUNNING)
            elif isinstance(event, RunCancelled):
                self._transition_to(RunState.CANCELLED)
            elif isinstance(event, RunCompleted):
                self._transition_to(RunState.COMPLETED)
            elif isinstance(event, RunFailed):
                self._transition_to(RunState.ERROR)

    def _transition_to(self, new_state: RunState) -> None:
        """Cambia de estado y emite RunStateChanged."""
        prev = self._state
        self._state = new_state
        if prev is not new_state:
            self._queue.put(RunStateChanged(previous=prev, current=new_state))
