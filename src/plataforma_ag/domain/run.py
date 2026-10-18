"""Modelos de una corrida: individuos, generaciones, historial y bitácora (ENT-007..ENT-012)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from plataforma_ag.domain.config import ExperimentConfig
from plataforma_ag.domain.enums import LogLevel, RunState, StopReason

Gene = int | float


@dataclass(frozen=True, slots=True)
class Individual:
    """Miembro de la población en una generación (ENT-007).

    AMB-007: se conservan todos los datos del individuo — f(x) crudo y aptitud
    transformada — para que el CSV pueda exportarlos completos.
    """

    individual_id: int
    genotype: tuple[Gene, ...]
    phenotype: tuple[float, ...]
    fitness: float  # f(x) crudo; ±inf si penalizado (RNF-03)
    aptitude: float | None = None  # aptitud interna transformada (RF-04)
    is_elite: bool = False


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    """Instantánea de una generación con estadísticas (ENT-008)."""

    generation: int
    best_fitness: float
    mean_fitness: float
    elite_fitness_cumulative: float
    individuals: tuple[Individual, ...] = ()


@dataclass(frozen=True, slots=True)
class ProgressSnapshot:
    """Estado efímero que alimenta la barra de carga (ENT-012, RF-18)."""

    current_generation: int
    max_generations: int
    percent: float
    state: RunState = RunState.IDLE


@dataclass(frozen=True, slots=True)
class LogEvent:
    """Entrada de la consola de bitácora (ENT-010, RF-19)."""

    timestamp: datetime
    level: LogLevel
    message: str
    generation: int | None = None


@dataclass(slots=True)
class RunHistory:
    """Resultado completo de una corrida (ENT-009). Mutable: crece por generación."""

    run_id: uuid.UUID = field(default_factory=uuid.uuid4)
    state: RunState = RunState.IDLE
    stop_reason: StopReason | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    loaded_from_csv: bool = False
    config: ExperimentConfig | None = None
    generations: list[GenerationRecord] = field(default_factory=list)
    log: list[LogEvent] = field(default_factory=list)
