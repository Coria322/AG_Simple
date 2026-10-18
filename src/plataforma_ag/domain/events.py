"""Eventos de dominio inmutables (ARCHITECTURE.md § Eventos y sincronización).

Viajan del hilo de trabajo a la GUI por una ``queue.Queue`` (RNF-01), por eso son
``frozen``: ningún consumidor puede alterarlos.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from plataforma_ag.domain.enums import LogLevel, RunState, StopReason
from plataforma_ag.domain.run import GenerationRecord


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True, kw_only=True)
class DomainEvent:
    """Base de todos los eventos; registra el instante de creación."""

    occurred_at: datetime = field(default_factory=_utc_now)


# --- Ciclo de vida de la corrida (RF-17, RF-18) -----------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class RunStarted(DomainEvent):
    run_id: uuid.UUID
    max_generations: int


@dataclass(frozen=True, slots=True, kw_only=True)
class RunPaused(DomainEvent):
    generation: int


@dataclass(frozen=True, slots=True, kw_only=True)
class RunResumed(DomainEvent):
    generation: int


@dataclass(frozen=True, slots=True, kw_only=True)
class RunCancelled(DomainEvent):
    generation: int


@dataclass(frozen=True, slots=True, kw_only=True)
class RunCompleted(DomainEvent):
    generation: int
    stop_reasons: tuple[StopReason, ...]  # RF-09: se registran ambos si coinciden


@dataclass(frozen=True, slots=True, kw_only=True)
class RunFailed(DomainEvent):
    message: str
    error_type: str = "Exception"


@dataclass(frozen=True, slots=True, kw_only=True)
class RunStateChanged(DomainEvent):
    previous: RunState
    current: RunState


# --- Telemetría (RF-18..RF-21) -----------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class GenerationCompleted(DomainEvent):
    generation: int
    snapshot: GenerationRecord


@dataclass(frozen=True, slots=True, kw_only=True)
class ProgressChanged(DomainEvent):
    current: int
    maximum: int
    percent: float


@dataclass(frozen=True, slots=True, kw_only=True)
class LogRecorded(DomainEvent):
    level: LogLevel
    message: str
    generation: int | None = None


# --- Presentación (PROV-UI-01) -----------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class ThemeChanged(DomainEvent):
    name: str
