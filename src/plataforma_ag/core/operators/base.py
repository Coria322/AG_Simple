"""Contratos Strategy de los operadores genéticos (RNF-04, ARCHITECTURE § Interfaces internas).

Cada operador expone ``name``, ``kind``, ``compatible_with`` y ``params_schema`` y
el método propio de su familia. Agregar un operador = escribir una clase nueva y
registrarla; ni el motor ni la UI cambian.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from plataforma_ag.domain.enums import ChromosomeType, OperatorKind
from plataforma_ag.domain.errors import ValidationError

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.intp]
Genotype = NDArray[Any]  # bits (uint8), reales (float64) o permutación (intp)

ALL_CHROMOSOME_TYPES: frozenset[ChromosomeType] = frozenset(ChromosomeType)


# --- Esquema de parámetros ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """Describe un parámetro de operador; la GUI lo usa para crear campos dinámicos (RF-12).

    ``minimum``/``maximum`` son opcionales; ``*_inclusive`` indica si el extremo es válido.
    """

    name: str
    type: type[int] | type[float]
    default: float
    label: str = ""
    minimum: float | None = None
    maximum: float | None = None
    min_inclusive: bool = True
    max_inclusive: bool = True

    def validate(self, value: Any) -> int | float:
        """Convierte y valida ``value``; lanza :class:`ValidationError` si no cumple."""
        if isinstance(value, bool):
            raise ValidationError("debe ser numérico", field=self.name)
        try:
            number = self.type(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"formato inválido: {value!r}", field=self.name) from exc
        if self.type is int and number != value:
            raise ValidationError("debe ser entero", field=self.name)
        if not np.isfinite(number):
            raise ValidationError("debe ser finito", field=self.name)
        if self.minimum is not None:
            too_low = number < self.minimum if self.min_inclusive else number <= self.minimum
            if too_low:
                sign = "≥" if self.min_inclusive else ">"
                raise ValidationError(f"debe ser {sign} {self.minimum}", field=self.name)
        if self.maximum is not None:
            too_high = number > self.maximum if self.max_inclusive else number >= self.maximum
            if too_high:
                sign = "≤" if self.max_inclusive else "<"
                raise ValidationError(f"debe ser {sign} {self.maximum}", field=self.name)
        return number


def resolve_params(
    schema: tuple[ParamSpec, ...], params: Mapping[str, Any] | None = None
) -> dict[str, int | float]:
    """Completa con valores por defecto, valida y rechaza parámetros desconocidos."""
    params = dict(params or {})
    known = {spec.name for spec in schema}
    unknown = sorted(set(params) - known)
    if unknown:
        raise ValidationError(f"parámetros desconocidos: {', '.join(unknown)}")
    return {spec.name: spec.validate(params.get(spec.name, spec.default)) for spec in schema}


# --- Contextos ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MutationContext:
    """Información de la corrida que necesita una mutación (RF-15)."""

    generation: int
    max_generations: int
    p_mutation: float
    lower: FloatArray
    upper: FloatArray


@dataclass(frozen=True, slots=True)
class Population:
    """Población en forma vectorizada: una fila por individuo.

    ``aptitude`` es la aptitud interna (siempre se maximiza, RF-04).
    """

    genotypes: Genotype
    aptitude: FloatArray

    def __len__(self) -> int:
        return int(self.genotypes.shape[0])


@dataclass(frozen=True, slots=True)
class ReplacementContext:
    """Datos para formar la nueva población (RF-16). Se ampliará en TASK-016."""

    generation: int
    rng: np.random.Generator
    params: Mapping[str, Any]


# --- Protocols ---------------------------------------------------------------


@runtime_checkable
class GeneticOperator(Protocol):
    """Atributos comunes de toda estrategia."""

    name: str
    kind: OperatorKind
    compatible_with: frozenset[ChromosomeType]
    params_schema: tuple[ParamSpec, ...]


@runtime_checkable
class SelectionOperator(GeneticOperator, Protocol):
    """Devuelve los índices de ``n`` padres a partir de la aptitud (RF-13)."""

    def select(
        self, fitness: FloatArray, n: int, rng: np.random.Generator, **params: Any
    ) -> IndexArray: ...


@runtime_checkable
class CrossoverOperator(GeneticOperator, Protocol):
    """Cruza dos padres y devuelve dos hijos (RF-14)."""

    def cross(
        self, p1: Genotype, p2: Genotype, rng: np.random.Generator, **params: Any
    ) -> tuple[Genotype, Genotype]: ...


@runtime_checkable
class MutationOperator(GeneticOperator, Protocol):
    """Devuelve una copia mutada del individuo (RF-15)."""

    def mutate(
        self, ind: Genotype, ctx: MutationContext, rng: np.random.Generator, **params: Any
    ) -> Genotype: ...


@runtime_checkable
class ReplacementScheme(GeneticOperator, Protocol):
    """Forma la población de la siguiente generación (RF-16)."""

    def apply(
        self, pop: Population, offspring: Population, ctx: ReplacementContext
    ) -> Population: ...


REQUIRED_METHOD: Mapping[OperatorKind, str] = {
    OperatorKind.SELECTION: "select",
    OperatorKind.CROSSOVER: "cross",
    OperatorKind.MUTATION: "mutate",
    OperatorKind.REPLACEMENT: "apply",
}
"""Método obligatorio según la familia del operador."""
