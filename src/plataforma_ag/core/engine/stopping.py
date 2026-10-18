"""Criterios de parada del algoritmo genético (RF-09, AMB-009).

Soporta parada por:
  - Máximo de generaciones Gmax (obligatorio en la GUI).
  - Estancamiento: |f_t - f_{t-K}| <= epsilon durante K generaciones consecutivas.
Ambos criterios pueden dispararse individualmente o de forma combinada.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from plataforma_ag.domain.config import StoppingCriteria
from plataforma_ag.domain.enums import StopReason


@dataclass(frozen=True, slots=True)
class StoppingEvaluator:
    """Evalúa las condiciones de terminación en cada generación."""

    criteria: StoppingCriteria

    def evaluate(
        self, generation: int, best_fitness_history: Sequence[float]
    ) -> tuple[bool, tuple[StopReason, ...]]:
        """Determina si la corrida debe detenerse.

        Args:
            generation: número de la generación completada.
            best_fitness_history: historial de mejores fitness crudos por generación.

        Returns:
            (debe_parar, razones): tupla con bandera y tupla de razones StopReason.
        """
        reasons: list[StopReason] = []

        # 1. Criterio de máximo de generaciones (Gmax)
        if generation >= self.criteria.max_generations:
            reasons.append(StopReason.MAX_GENERATIONS)

        # 2. Criterio de estancamiento
        if self.criteria.stagnation_enabled:
            k = self.criteria.stagnation_generations
            eps = self.criteria.stagnation_epsilon
            n_history = len(best_fitness_history)
            if n_history > k:
                # Comprobar |f_t - f_{t-k}|
                f_t = best_fitness_history[-1]
                f_prev = best_fitness_history[-1 - k]
                if np.isfinite(f_t) and np.isfinite(f_prev) and abs(f_t - f_prev) <= eps:
                    reasons.append(StopReason.STAGNATION)

        if reasons:
            return True, tuple(reasons)
        return False, ()
