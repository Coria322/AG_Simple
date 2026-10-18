"""Hilo de trabajo y cola de eventos para la ejecución del AG (RNF-01, RF-17, RF-18).

El ciclo del AG corre en un hilo secundario y publica eventos inmutables en una
``queue.Queue``. El hilo nunca toca widgets ni interactúa con la interfaz gráfica.
La pausa se gestiona con ``threading.Event`` y la cancelación de forma cooperativa
por generación (con tiempo de parada <= 1 segundo).
"""

from __future__ import annotations

import queue
import threading
import uuid

from plataforma_ag.core.engine.ga_engine import GAEngine
from plataforma_ag.domain.events import (
    DomainEvent,
    GenerationCompleted,
    ProgressChanged,
    RunCancelled,
    RunCompleted,
    RunFailed,
    RunPaused,
    RunResumed,
    RunStarted,
)


class WorkerThread(threading.Thread):
    """Hilo de trabajo que ejecuta GAEngine y comunica telemetría vía cola de eventos."""

    def __init__(
        self,
        engine: GAEngine,
        event_queue: queue.Queue[DomainEvent],
        run_id: uuid.UUID | None = None,
    ) -> None:
        super().__init__(name="GAWorkerThread", daemon=True)
        self.engine = engine
        self.event_queue = event_queue
        self.run_id = run_id or uuid.uuid4()

        self._pause_event = threading.Event()
        self._pause_event.set()  # Inicia activo (no pausado)

        self._cancel_event = threading.Event()
        self.current_generation = 0

    def pause(self) -> None:
        """Pausa la ejecución entre generaciones."""
        if self._pause_event.is_set():
            self._pause_event.clear()

    def resume(self) -> None:
        """Reanuda la ejecución tras una pausa."""
        if not self._pause_event.is_set():
            self._pause_event.set()
            self.event_queue.put(RunResumed(generation=self.current_generation))

    def cancel(self) -> None:
        """Solicita la cancelación cooperativa inmediata de la corrida."""
        self._cancel_event.set()
        # Si estaba en pausa, liberarlo para que pueda salir limpiamente
        self._pause_event.set()

    def run(self) -> None:
        """Ciclo principal de ejecución en el hilo de trabajo."""
        max_gen = max(self.engine.config.stopping.max_generations, 1)
        self.event_queue.put(RunStarted(run_id=self.run_id, max_generations=max_gen))

        try:
            for record, stop_reasons in self.engine.run():
                self.current_generation = record.generation

                # Comprobar cancelación cooperativa
                if self._cancel_event.is_set():
                    self.event_queue.put(RunCancelled(generation=self.current_generation))
                    return

                # Publicar instantánea de generación
                self.event_queue.put(
                    GenerationCompleted(generation=self.current_generation, snapshot=record)
                )

                # Publicar progreso
                pct = min(100.0, (self.current_generation / max_gen) * 100.0)
                self.event_queue.put(
                    ProgressChanged(
                        current=self.current_generation, maximum=max_gen, percent=pct
                    )
                )

                # Si se alcanzaron criterios de parada
                if stop_reasons is not None:
                    self.event_queue.put(
                        RunCompleted(
                            generation=self.current_generation, stop_reasons=stop_reasons
                        )
                    )
                    return

                # Esperar si el hilo fue pausado
                if not self._pause_event.is_set():
                    self.event_queue.put(RunPaused(generation=self.current_generation))
                    self._pause_event.wait()

                # Comprobar si se canceló mientras estaba en pausa
                if self._cancel_event.is_set():
                    self.event_queue.put(RunCancelled(generation=self.current_generation))
                    return

        except Exception as exc:
            # Fallo no controlado: publicar RunFailed para que la app no muera (RNF-01)
            self.event_queue.put(
                RunFailed(message=str(exc), error_type=type(exc).__name__)
            )
