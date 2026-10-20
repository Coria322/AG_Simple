"""Operadores de selección genética (RF-13, SUP-013, AMB-010, AMB-012).

Implementa:
  - ROULETTE: Ruleta proporcional a la aptitud.
  - STOCHASTIC_UNIVERSAL_SAMPLING: Muestreo universal estocástico con punteros equidistantes.
  - LINEAR_RANKING: Ranking lineal con presión selectiva s en [1.0, 2.0] (default 1.5).
  - NONLINEAR_RANKING: Ranking no lineal con parámetro q en (0.0, 1.0).
  - TOURNAMENT: Torneo determinista de tamaño k >= 2.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from plataforma_ag.core.operators.base import (
    ALL_CHROMOSOME_TYPES,
    FloatArray,
    IndexArray,
    ParamSpec,
)
from plataforma_ag.core.operators.registry import register_operator
from plataforma_ag.domain.enums import ChromosomeType, OperatorKind, SelectionMethod

NO_PERMUTATION = frozenset(
    {
        ChromosomeType.BINARY,
        ChromosomeType.GRAY,
        ChromosomeType.REAL,
    }
)


def _normalize_probabilities(probs: FloatArray) -> FloatArray:
    """Garantiza que las probabilidades sumen 1.0 exactamente dentro de tolerancias numéricas."""
    total = float(np.sum(probs))
    if total <= 0.0 or not np.isfinite(total):
        # Distribución uniforme si la suma es nula o inválida
        n = len(probs)
        return np.full(n, 1.0 / n, dtype=np.float64)
    norm = probs / total
    # Corrección de redondeo residual en el último elemento
    diff = 1.0 - float(np.sum(norm))
    norm[-1] += diff
    return norm


@register_operator
class RouletteSelection:
    """Selección proporcional por ruleta."""

    name = SelectionMethod.ROULETTE.value
    kind = OperatorKind.SELECTION
    compatible_with = ALL_CHROMOSOME_TYPES
    params_schema = ()

    def select(
        self, fitness: FloatArray, n: int, rng: np.random.Generator, **params: Any
    ) -> IndexArray:
        pop_size = len(fitness)
        probs = _normalize_probabilities(np.maximum(fitness, 0.0))
        return rng.choice(pop_size, size=n, p=probs, replace=True)


@register_operator
class SUSSelection:
    """Stochastic Universal Sampling (SUS) con n punteros equidistantes."""

    name = SelectionMethod.STOCHASTIC_UNIVERSAL_SAMPLING.value
    kind = OperatorKind.SELECTION
    compatible_with = ALL_CHROMOSOME_TYPES
    params_schema = ()

    def select(
        self, fitness: FloatArray, n: int, rng: np.random.Generator, **params: Any
    ) -> IndexArray:
        pop_size = len(fitness)
        safe_fitness = np.maximum(fitness, 0.0)
        total_fitness = float(np.sum(safe_fitness))

        if total_fitness <= 0.0 or not np.isfinite(total_fitness):
            return rng.integers(0, pop_size, size=n, dtype=np.intp)

        step = total_fitness / n
        start = rng.uniform(0.0, step)
        pointers = start + np.arange(n) * step

        # Suma acumulativa para encontrar los individuos
        cum_sum = np.cumsum(safe_fitness)
        selected = np.searchsorted(cum_sum, pointers)
        return np.clip(selected, 0, pop_size - 1)


@register_operator
class LinearRankingSelection:
    """Ranking lineal con parámetro de presión selectiva s en [1.0, 2.0] (SUP-013)."""

    name = SelectionMethod.LINEAR_RANKING.value
    kind = OperatorKind.SELECTION
    compatible_with = ALL_CHROMOSOME_TYPES
    params_schema = (
        ParamSpec("s", float, 1.5, label="Presión selectiva (s)", minimum=1.0, maximum=2.0),
    )

    def select(
        self, fitness: FloatArray, n: int, rng: np.random.Generator, **params: Any
    ) -> IndexArray:
        pop_size = len(fitness)
        s = float(params.get("s", 1.5))

        # Ordenar individuos de peor (rank 0) a mejor (rank Np-1)
        # Valores penalizados (0.0 o <=0) quedan en los peores puestos
        order = np.argsort(fitness)

        if pop_size == 1:
            return np.zeros(n, dtype=np.intp)

        # Fórmula de Palma-Marín / Whitley:
        # P(i) = (2 - s) / Np + 2 * i * (s - 1) / (Np * (Np - 1))
        ranks = np.arange(pop_size, dtype=np.float64)
        term1 = (2.0 - s) / pop_size
        term2 = (2.0 * ranks * (s - 1.0)) / (pop_size * (pop_size - 1))
        rank_probs = term1 + term2

        # Asignar probabilidades a las posiciones originales
        probs = np.empty(pop_size, dtype=np.float64)
        probs[order] = rank_probs
        norm_probs = _normalize_probabilities(probs)

        return rng.choice(pop_size, size=n, p=norm_probs, replace=True)


@register_operator
class NonLinearRankingSelection:
    """Ranking no lineal exponencial con parámetro q en (0.0, 1.0)."""

    name = SelectionMethod.NONLINEAR_RANKING.value
    kind = OperatorKind.SELECTION
    compatible_with = NO_PERMUTATION  # Deshabilitado en Permutaciones según matriz RF-12
    params_schema = (
        ParamSpec(
            "q",
            float,
            0.05,
            label="Factor de probabilidad (q)",
            minimum=0.0,
            maximum=1.0,
            min_inclusive=False,
            max_inclusive=False,
        ),
    )

    def select(
        self, fitness: FloatArray, n: int, rng: np.random.Generator, **params: Any
    ) -> IndexArray:
        pop_size = len(fitness)
        q = float(params.get("q", 0.05))

        # Ordenar de mejor (rank 0) a peor (rank Np-1)
        order = np.argsort(-fitness)

        ranks = np.arange(pop_size, dtype=np.float64)
        rank_probs = q * ((1.0 - q) ** ranks)

        probs = np.empty(pop_size, dtype=np.float64)
        probs[order] = rank_probs
        norm_probs = _normalize_probabilities(probs)

        return rng.choice(pop_size, size=n, p=norm_probs, replace=True)


@register_operator
class TournamentSelection:
    """Selección por torneo determinista de tamaño k >= 2."""

    name = SelectionMethod.TOURNAMENT.value
    kind = OperatorKind.SELECTION
    compatible_with = ALL_CHROMOSOME_TYPES
    params_schema = (ParamSpec("k", int, 3, label="Tamaño de torneo (k)", minimum=2),)

    def select(
        self, fitness: FloatArray, n: int, rng: np.random.Generator, **params: Any
    ) -> IndexArray:
        pop_size = len(fitness)
        k = int(params.get("k", 3))
        selected = np.empty(n, dtype=np.intp)

        for i in range(n):
            candidates = rng.integers(0, pop_size, size=k)
            cand_fitness = fitness[candidates]
            best_idx = int(np.argmax(cand_fitness))
            selected[i] = candidates[best_idx]

        return selected
