"""Dataclasses de configuración de un experimento (DOMAIN_SCHEMA ENT-001..ENT-006).

Son inmutables (``frozen``): para modificar se usa :func:`dataclasses.replace`.
Las reglas de validación de rangos viven en ``domain.validation`` (TASK-007/036);
aquí solo se modelan la forma y los valores por defecto del ERS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from plataforma_ag.domain.enums import (
    AngleUnit,
    BenchmarkFunction,
    ChromosomeType,
    CrossoverMethod,
    FitnessTransform,
    MutationMethod,
    ObjectiveSense,
    RedundantSpaceStrategy,
    ReplacementMethod,
    SelectionMethod,
)

# Valores por defecto del ERS (RF-07, RF-08, RF-09) y precarga de demostración (RF-05).
DEFAULT_POPULATION_SIZE = 100
DEFAULT_P_CROSSOVER = 0.85
DEFAULT_P_MUTATION = 0.05
DEFAULT_MAX_GENERATIONS = 200
DEFAULT_STAGNATION_GENERATIONS = 25
DEFAULT_STAGNATION_EPSILON = 1e-6
DEFAULT_DIMENSIONS = 2
DEFAULT_BITS = 16
DEMO_BOUND_MIN = -5.12
DEMO_BOUND_MAX = 5.12
MIN_BITS = 1
MAX_BITS = 64  # AMB-002: tope confirmado


def variable_name(index: int) -> str:
    """Nombre de la variable de la dimensión ``index`` (base 0): ``x1``, ``x2``, …"""
    return f"x{index + 1}"


@dataclass(frozen=True, slots=True)
class VariableBound:
    """Dominio de una dimensión (ENT-003, RF-05)."""

    variable: str
    min: float = DEMO_BOUND_MIN
    max: float = DEMO_BOUND_MAX
    bits: int = DEFAULT_BITS  # solo BINARY/GRAY


def _demo_bounds(dimensions: int = DEFAULT_DIMENSIONS) -> tuple[VariableBound, ...]:
    return tuple(VariableBound(variable=variable_name(i)) for i in range(dimensions))


@dataclass(frozen=True, slots=True)
class ProblemDefinition:
    """Función objetivo, sentido y dominios (ENT-002, RF-01..RF-06)."""

    benchmark: BenchmarkFunction = BenchmarkFunction.CUSTOM_EXPRESSION
    dimensions: int = DEFAULT_DIMENSIONS
    expression: str | None = None
    angle_unit: AngleUnit = AngleUnit.RADIANS
    objective: ObjectiveSense = ObjectiveSense.MINIMIZE
    redundant_space_strategy: RedundantSpaceStrategy = RedundantSpaceStrategy.SATURATION
    fitness_transform: FitnessTransform = FitnessTransform.NEGATE_PLUS_C  # AMB-001
    bounds: tuple[VariableBound, ...] = field(default_factory=_demo_bounds)


@dataclass(frozen=True, slots=True)
class GeneticParameters:
    """Población, representación y probabilidades (ENT-004, RF-07, RF-08, RF-11)."""

    population_size: int = DEFAULT_POPULATION_SIZE
    chromosome_type: ChromosomeType = ChromosomeType.BINARY  # AMB-003: 4 opciones
    p_crossover: float = DEFAULT_P_CROSSOVER
    p_mutation: float = DEFAULT_P_MUTATION
    seed: int | None = None  # AMB-008


@dataclass(frozen=True, slots=True)
class OperatorConfig:
    """Operadores elegidos y sus parámetros (ENT-005, RF-12..RF-16)."""

    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    selection_params: dict[str, Any] = field(default_factory=dict)
    crossover_method: CrossoverMethod = CrossoverMethod.SINGLE_POINT
    crossover_params: dict[str, Any] = field(default_factory=dict)
    mutation_method: MutationMethod = MutationMethod.SIMPLE_FLIP
    mutation_params: dict[str, Any] = field(default_factory=dict)
    replacement_method: ReplacementMethod = ReplacementMethod.GENERATIONAL
    n_replace: int | None = None
    elitism_count: int | None = None  # AMB-005: combinable con el esquema principal
    crowding_factor: int | None = None


@dataclass(frozen=True, slots=True)
class StoppingCriteria:
    """Criterios de parada (ENT-006, RF-09). AMB-009: Gmax siempre obligatorio."""

    max_generations: int = DEFAULT_MAX_GENERATIONS
    stagnation_enabled: bool = False
    stagnation_generations: int = DEFAULT_STAGNATION_GENERATIONS
    stagnation_epsilon: float = DEFAULT_STAGNATION_EPSILON


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Raíz de la configuración de un experimento (ENT-001, RF-22)."""

    project_name: str = "Experimento AG"
    timestamp: datetime = field(default_factory=_utc_now)
    schema_version: str = "1"
    problem: ProblemDefinition = field(default_factory=ProblemDefinition)
    genetic: GeneticParameters = field(default_factory=GeneticParameters)
    operators: OperatorConfig = field(default_factory=OperatorConfig)
    stopping: StoppingCriteria = field(default_factory=StoppingCriteria)
