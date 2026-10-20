"""Servicio de bitácora y registro de eventos de la corrida (RF-19, TASK-039).

Registra con marcas de tiempo en milisegundos (`[HH:MM:SS.mmm]`):
  - Avance generacional con mejor fitness local, global acumulado y promedio.
  - Advertencias numéricas (penalizaciones, desbordamientos, etc.).
  - Criterios de paro detonados.
  - Errores críticos.

Mantiene un búfer circular en memoria para evitar consumo ilimitado de RAM en corridas largas.
"""

from __future__ import annotations

import threading
from collections import deque
from datetime import UTC, datetime

from plataforma_ag.domain.enums import LogLevel, StopReason
from plataforma_ag.domain.events import LogRecorded
from plataforma_ag.domain.run import GenerationRecord


def format_timestamp(dt: datetime | None = None) -> str:
    """Devuelve la marca de tiempo en formato [HH:MM:SS.mmm]."""
    if dt is None:
        dt = datetime.now(UTC)
    return dt.strftime("[%H:%M:%S.") + f"{dt.microsecond // 1000:03d}]"


class RunLogger:
    """Bitácora en memoria con búfer acotado y formato estandarizado."""

    def __init__(self, max_entries: int = 5000) -> None:
        self.max_entries = max_entries
        self._entries: deque[LogRecorded] = deque(maxlen=max_entries)
        self._lock = threading.Lock()

    def log_generation(
        self, record: GenerationRecord, cumulative_elite: float | None = None
    ) -> LogRecorded:
        """Registra el resumen de una generación completada."""
        elite_val = record.best_fitness if cumulative_elite is None else cumulative_elite
        msg = (
            f"Gen {record.generation:03d} | "
            f"Mejor local: {record.best_fitness:.6g} | "
            f"Mejor global: {elite_val:.6g} | "
            f"Promedio: {record.mean_fitness:.6g}"
        )
        return self._append(LogLevel.INFO, msg, record.generation)

    def log_stop(self, reasons: tuple[StopReason, ...], generation: int) -> LogRecorded:
        """Registra la activación de uno o varios criterios de parada."""
        reasons_str = ", ".join(r.value for r in reasons)
        msg = f"Criterio de parada alcanzado en generación {generation}: {reasons_str}"
        return self._append(LogLevel.INFO, msg, generation)

    def log_warning(self, message: str, generation: int | None = None) -> LogRecorded:
        """Registra una advertencia numérica o de configuración."""
        prefix = f"[Gen {generation}] " if generation is not None else ""
        return self._append(LogLevel.WARN, f"{prefix}{message}", generation)

    def log_error(self, message: str, generation: int | None = None) -> LogRecorded:
        """Registra un error crítico."""
        prefix = f"[Gen {generation}] " if generation is not None else ""
        return self._append(LogLevel.ERROR, f"{prefix}{message}", generation)

    def log_info(self, message: str, generation: int | None = None) -> LogRecorded:
        """Registra un mensaje informativo general."""
        prefix = f"[Gen {generation}] " if generation is not None else ""
        return self._append(LogLevel.INFO, f"{prefix}{message}", generation)

    def get_entries(self) -> list[LogRecorded]:
        """Devuelve una copia de las entradas acumuladas en la bitácora."""
        with self._lock:
            return list(self._entries)

    def clear(self) -> None:
        """Limpia el búfer de bitácora."""
        with self._lock:
            self._entries.clear()

    def _append(self, level: LogLevel, message: str, generation: int | None) -> LogRecorded:
        """Crea y almacena el evento LogRecorded thread-safe."""
        event = LogRecorded(
            level=level,
            message=message,
            generation=generation,
        )
        with self._lock:
            self._entries.append(event)
        return event
