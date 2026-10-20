"""Controlador de telemetría y bitácora (RF-18..RF-21, TASK-048..051).

Recibe eventos del motor, alimenta el almacén de historia y bitácora,
y provee datos formateados a la gráfica, tabla de población y consola.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

from plataforma_ag.domain.enums import RunState
from plataforma_ag.domain.events import (
    DomainEvent,
    GenerationCompleted,
    LogRecorded,
    ProgressChanged,
    RunCancelled,
    RunCompleted,
    RunFailed,
    RunStarted,
)
from plataforma_ag.domain.run import GenerationRecord, Individual, ProgressSnapshot
from plataforma_ag.services.history import ConvergenceSeries, RunHistoryStore
from plataforma_ag.services.run_logger import RunLogger


class TelemetryController:
    """Controlador que intermedia entre los datos de telemetría y las vistas."""

    def __init__(
        self,
        history_store: RunHistoryStore | None = None,
        logger: RunLogger | None = None,
    ) -> None:
        self.history_store = history_store or RunHistoryStore()
        self.logger = logger or RunLogger()

        self._on_progress_listeners: list[Callable[[ProgressSnapshot], None]] = []
        self._on_generation_listeners: list[Callable[[GenerationRecord, float], None]] = []
        self._on_log_listeners: list[Callable[[LogRecorded], None]] = []
        self._on_run_finished_listeners: list[Callable[[str], None]] = []

    def reset(self) -> None:
        """Reinicia el historial y la bitácora para una nueva corrida."""
        self.history_store.clear()
        self.logger.clear()

    def add_progress_listener(self, cb: Callable[[ProgressSnapshot], None]) -> None:
        self._on_progress_listeners.append(cb)

    def add_generation_listener(self, cb: Callable[[GenerationRecord, float], None]) -> None:
        self._on_generation_listeners.append(cb)

    def add_log_listener(self, cb: Callable[[LogRecorded], None]) -> None:
        self._on_log_listeners.append(cb)

    def add_run_finished_listener(self, cb: Callable[[str], None]) -> None:
        self._on_run_finished_listeners.append(cb)

    def handle_event(self, event: DomainEvent) -> None:
        """Procesa un evento de dominio generado por el motor en ejecución."""
        if isinstance(event, GenerationCompleted):
            record = event.snapshot
            self.history_store.add_generation(record)
            cumulative_elite = (
                self.history_store.global_best.fitness
                if self.history_store.global_best
                and self.history_store.global_best.fitness is not None
                else record.best_fitness
            )
            log_ev = self.logger.log_generation(record, cumulative_elite=cumulative_elite)
            for log_cb in self._on_log_listeners:
                with contextlib.suppress(Exception):
                    log_cb(log_ev)
            for gen_cb in self._on_generation_listeners:
                with contextlib.suppress(Exception):
                    gen_cb(record, cumulative_elite)

        elif isinstance(event, ProgressChanged):
            snap = ProgressSnapshot(
                current_generation=event.current,
                max_generations=event.maximum,
                percent=event.percent,
                state=RunState.RUNNING,
            )
            for prog_cb in self._on_progress_listeners:
                with contextlib.suppress(Exception):
                    prog_cb(snap)

        elif isinstance(event, LogRecorded):
            self.logger.add_event(event)
            for log_cb in self._on_log_listeners:
                with contextlib.suppress(Exception):
                    log_cb(event)

        elif isinstance(event, RunStarted):
            self.reset()
            log_ev = self.logger.log_info(
                f"Corrida iniciada: ID={event.run_id}, Gmax={event.max_generations}"
            )
            for log_cb in self._on_log_listeners:
                with contextlib.suppress(Exception):
                    log_cb(log_ev)

        elif isinstance(event, RunCompleted):
            log_ev = self.logger.log_stop(event.stop_reasons, generation=event.generation)
            for log_cb in self._on_log_listeners:
                with contextlib.suppress(Exception):
                    log_cb(log_ev)
            for fin_cb in self._on_run_finished_listeners:
                with contextlib.suppress(Exception):
                    fin_cb("completed")

        elif isinstance(event, RunCancelled):
            log_ev = self.logger.log_warning(
                f"Corrida cancelada por el usuario en la generación {event.generation}",
                generation=event.generation,
            )
            for log_cb in self._on_log_listeners:
                with contextlib.suppress(Exception):
                    log_cb(log_ev)
            for fin_cb in self._on_run_finished_listeners:
                with contextlib.suppress(Exception):
                    fin_cb("cancelled")

        elif isinstance(event, RunFailed):
            log_ev = self.logger.log_error(f"Error ({event.error_type}): {event.message}")
            for log_cb in self._on_log_listeners:
                with contextlib.suppress(Exception):
                    log_cb(log_ev)
            for fin_cb in self._on_run_finished_listeners:
                with contextlib.suppress(Exception):
                    fin_cb("error")

    def get_population_page(
        self,
        generation: int,
        page: int = 1,
        page_size: int = 25,
        sort_by: str = "id",
        ascending: bool = True,
    ) -> tuple[tuple[Individual, ...], int]:
        """Obtiene una página de individuos para una generación dada."""
        return self.history_store.get_population_page(
            generation=generation,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            ascending=ascending,
        )

    def get_evolution_series(self) -> ConvergenceSeries:
        """Obtiene las series históricas de fitness para el gráfico de convergencia."""
        return self.history_store.get_convergence_series()

    def get_log_entries(self) -> list[LogRecorded]:
        return self.logger.get_entries()
