"""Función objetivo y transformaciones de aptitud (RF-04, AMB-001, RNF-03).

El motor del AG siempre MAXIMIZA internamente sobre la aptitud (*aptitude*).
Para MINIMIZAR se aplica una transformación de aptitud intercambiable:
  - ``NEGATE_PLUS_C`` (por defecto, AMB-001): aptitude = -f + C, con C = max(f) de la generación.
  - ``INVERSE_SHIFT``: aptitude = 1 / (1 + f - f_min), con f_min = min(f) de la generación.

El valor reportado al usuario y almacenado en el historial SIEMPRE es f(x) crudo.
Individuos con fallo matemático (±inf, nan) reciben aptitud 0.0 (AMB-010).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from plataforma_ag.domain import DTYPE
from plataforma_ag.domain.enums import FitnessTransform, ObjectiveSense

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.intp]


@runtime_checkable
class Evaluator(Protocol):
    """Protocolo de cualquier evaluador de puntos fenotípicos en R^N."""

    dimensions: int

    def evaluate(self, population: ArrayLike) -> FloatArray: ...


# --- Estrategias de transformación de aptitud (AMB-001) ----------------------


def transform_negate_plus_c(
    raw_fitness: FloatArray, sense: ObjectiveSense
) -> FloatArray:
    """Transformación -f + C (AMB-001).

    Para MINIMIZE: aptitud = -f + max(f).
    Si todos son idénticos, aptitud = 1.0 (uniforme no nula).
    Valores penalizados (no finitos) reciben aptitud 0.0 (AMB-010).
    """
    aptitude = np.zeros_like(raw_fitness, dtype=DTYPE)
    finite_mask = np.isfinite(raw_fitness)
    if not np.any(finite_mask):
        return aptitude  # todos penalizados: aptitud 0.0

    valid_f = raw_fitness[finite_mask]

    if sense is ObjectiveSense.MINIMIZE:
        c = float(np.max(valid_f))
        diff = c - valid_f
        # Si la población converge a un único valor
        if np.all(diff == 0.0):
            aptitude[finite_mask] = 1.0
        else:
            aptitude[finite_mask] = diff
    else:  # MAXIMIZE
        min_f = float(np.min(valid_f))
        if min_f < 0.0:
            aptitude[finite_mask] = valid_f - min_f
        else:
            aptitude[finite_mask] = valid_f
        # Si todos son 0.0 o idénticos
        if np.all(aptitude[finite_mask] == 0.0):
            aptitude[finite_mask] = 1.0

    return aptitude


def transform_inverse_shift(
    raw_fitness: FloatArray, sense: ObjectiveSense
) -> FloatArray:
    """Transformación 1 / (1 + f - f_min) (RF-04 alternativo).

    Para MINIMIZE: individuos con menor f reciben mayor aptitud (máximo 1.0).
    Valores penalizados reciben aptitud 0.0.
    """
    aptitude = np.zeros_like(raw_fitness, dtype=DTYPE)
    finite_mask = np.isfinite(raw_fitness)
    if not np.any(finite_mask):
        return aptitude

    valid_f = raw_fitness[finite_mask]

    if sense is ObjectiveSense.MINIMIZE:
        f_min = float(np.min(valid_f))
        diff = valid_f - f_min
        aptitude[finite_mask] = 1.0 / (1.0 + diff)
    else:  # MAXIMIZE
        f_max = float(np.max(valid_f))
        diff = f_max - valid_f
        aptitude[finite_mask] = 1.0 / (1.0 + diff)

    return aptitude


_TRANSFORM_REGISTRY: dict[
    FitnessTransform, Callable[[FloatArray, ObjectiveSense], FloatArray]
] = {
    FitnessTransform.NEGATE_PLUS_C: transform_negate_plus_c,
    FitnessTransform.INVERSE_SHIFT: transform_inverse_shift,
}


# --- Función Objetivo --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ObjectiveFunction:
    """Encapsula el evaluador, el sentido de optimización y la transformación de aptitud."""

    evaluator: Evaluator
    sense: ObjectiveSense = ObjectiveSense.MINIMIZE
    transform: FitnessTransform = FitnessTransform.NEGATE_PLUS_C  # AMB-001

    @property
    def dimensions(self) -> int:
        return self.evaluator.dimensions

    def evaluate(self, phenotypes: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Evalúa fenotipos y devuelve (raw_fitness, aptitude).

        raw_fitness: f(x) crudo de la función matemática.
        aptitude: valor no negativo a maximizar internamente por selección.
        """
        raw = self.evaluator.evaluate(phenotypes)
        raw_arr = np.asarray(raw, dtype=DTYPE)

        transform_fn = _TRANSFORM_REGISTRY.get(
            self.transform, transform_negate_plus_c
        )
        aptitude = transform_fn(raw_arr, self.sense)

        return raw_arr, aptitude

    def is_better(self, f1: float, f2: float) -> bool:
        """Determina si f1 es estrictamente mejor que f2 según el sentido."""
        if not np.isfinite(f1):
            return False
        if not np.isfinite(f2):
            return True
        if self.sense is ObjectiveSense.MINIMIZE:
            return f1 < f2
        return f1 > f2

    def rank_indices(self, raw_fitness: FloatArray) -> IndexArray:
        """Devuelve los índices de la población ordenados de MEJOR a PEOR individuo."""
        # Valores no finitos (inf, nan) van al final (peores)
        finite = np.isfinite(raw_fitness)
        order = np.argsort(raw_fitness)
        if self.sense is ObjectiveSense.MAXIMIZE:
            order = order[::-1]

        # Asegurar que los finitos vengan primero y los no finitos después
        finite_order = [idx for idx in order if finite[idx]]
        non_finite_order = [idx for idx in order if not finite[idx]]
        return np.array(finite_order + non_finite_order, dtype=np.intp)
