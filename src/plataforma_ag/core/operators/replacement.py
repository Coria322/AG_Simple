"""Esquemas de reemplazo generacional y de nicho (RF-16, AMB-005, AMB-012, SUP-013).

Implementa:
  - GENERATIONAL: Reemplazo generacional con elitismo opcional.
  - ELITISM: Esquema elitista explícito (conserva los E mejores de t en t+1).
  - STEADY_STATE: Estado estable con n_replace < Np (reemplaza a los n peores).
  - CROWDING: Reemplazo por apiñamiento (factor CF, distancia y sustitución solo si mejora).
"""

from __future__ import annotations

import numpy as np

from plataforma_ag.core.operators.base import (
    ALL_CHROMOSOME_TYPES,
    ParamSpec,
    Population,
    ReplacementContext,
)
from plataforma_ag.core.operators.registry import register_operator
from plataforma_ag.domain.enums import OperatorKind, ReplacementMethod
from plataforma_ag.domain.errors import ValidationError


def _distance(ind1: np.ndarray, ind2: np.ndarray) -> float:
    """Calcula la distancia fenotípica o genotípica (Hamming o Euclidiana)."""
    if np.issubdtype(ind1.dtype, np.integer):
        # Distancia de Hamming para cadenas discretas / bits / permutaciones
        return float(np.count_nonzero(ind1 != ind2))
    # Distancia euclidiana para vectores reales
    diff = ind1.astype(np.float64) - ind2.astype(np.float64)
    return float(np.linalg.norm(diff))


def _apply_elitism(
    pop: Population,
    offspring: Population,
    e_count: int,
) -> Population:
    """Conserva los e_count mejores de pop e inserta la descendencia en el resto."""
    pop_size = len(pop)
    e = min(max(e_count, 0), pop_size)
    if e == 0:
        return offspring

    # Índices ordenados de mejor a peor según aptitud
    pop_best_idx = np.argsort(-pop.aptitude)[:e]

    # Ordenar la descendencia de mejor a peor para tomar los (pop_size - e) mejores
    off_best_idx = np.argsort(-offspring.aptitude)[: pop_size - e]

    new_genotypes = np.concatenate(
        [
            pop.genotypes[pop_best_idx],
            offspring.genotypes[off_best_idx],
        ],
        axis=0,
    )

    new_aptitude = np.concatenate(
        [
            pop.aptitude[pop_best_idx],
            offspring.aptitude[off_best_idx],
        ],
        axis=0,
    )

    return Population(genotypes=new_genotypes, aptitude=new_aptitude)


@register_operator
class GenerationalReplacement:
    """Reemplazo generacional completo con soporte de elitismo opcional (AMB-005)."""

    name = ReplacementMethod.GENERATIONAL.value
    kind = OperatorKind.REPLACEMENT
    compatible_with = ALL_CHROMOSOME_TYPES
    params_schema = (
        ParamSpec(
            "elitism_count",
            int,
            0,
            label="Individuos élite (E)",
            minimum=0,
        ),
    )

    def apply(self, pop: Population, offspring: Population, ctx: ReplacementContext) -> Population:
        e = int(ctx.params.get("elitism_count", 0))
        return _apply_elitism(pop, offspring, e)


@register_operator
class ElitismReplacement:
    """Reemplazo con elitismo estricto E >= 1."""

    name = ReplacementMethod.ELITISM.value
    kind = OperatorKind.REPLACEMENT
    compatible_with = ALL_CHROMOSOME_TYPES
    params_schema = (
        ParamSpec(
            "elitism_count",
            int,
            2,
            label="Individuos élite (E)",
            minimum=1,
        ),
    )

    def apply(self, pop: Population, offspring: Population, ctx: ReplacementContext) -> Population:
        e = int(ctx.params.get("elitism_count", 2))
        if e < 1:
            raise ValidationError("Elitismo requiere E ≥ 1.", field="elitism_count")
        return _apply_elitism(pop, offspring, e)


@register_operator
class SteadyStateReplacement:
    """Estado estable: reemplaza a los n peores individuos de pop por los n descendientes."""

    name = ReplacementMethod.STEADY_STATE.value
    kind = OperatorKind.REPLACEMENT
    compatible_with = ALL_CHROMOSOME_TYPES
    params_schema = (
        ParamSpec(
            "n_replace",
            int,
            2,
            label="Individuos a reemplazar (n)",
            minimum=1,
        ),
    )

    def apply(self, pop: Population, offspring: Population, ctx: ReplacementContext) -> Population:
        pop_size = len(pop)
        n = int(ctx.params.get("n_replace", 2))
        if n >= pop_size:
            raise ValidationError(
                f"En estado estable n_replace debe ser < Np ({n} ≥ {pop_size}).",
                field="n_replace",
            )
        if n < 1:
            raise ValidationError("En estado estable n_replace debe ser ≥ 1.", field="n_replace")

        # Índices ordenados de peor a mejor en pop (para reemplazar a los n peores)
        pop_worst_idx = np.argsort(pop.aptitude)[:n]
        # Índices de los mejores descendientes
        off_best_idx = np.argsort(-offspring.aptitude)[:n]

        new_genotypes = pop.genotypes.copy()
        new_aptitude = pop.aptitude.copy()

        new_genotypes[pop_worst_idx] = offspring.genotypes[off_best_idx]
        new_aptitude[pop_worst_idx] = offspring.aptitude[off_best_idx]

        return Population(genotypes=new_genotypes, aptitude=new_aptitude)


@register_operator
class CrowdingReplacement:
    """Apiñamiento (Crowding): cada descendiente compite con el más similar en una muestra CF."""

    name = ReplacementMethod.CROWDING.value
    kind = OperatorKind.REPLACEMENT
    compatible_with = ALL_CHROMOSOME_TYPES
    params_schema = (
        ParamSpec(
            "crowding_factor",
            int,
            3,
            label="Factor de crowding (CF)",
            minimum=1,
        ),
    )

    def apply(self, pop: Population, offspring: Population, ctx: ReplacementContext) -> Population:
        pop_size = len(pop)
        cf = int(ctx.params.get("crowding_factor", 3))
        cf = min(max(cf, 1), pop_size)
        rng = ctx.rng

        new_genotypes = pop.genotypes.copy()
        new_aptitude = pop.aptitude.copy()

        # Para cada descendiente
        for i in range(len(offspring)):
            off_ind = offspring.genotypes[i]
            off_apt = offspring.aptitude[i]

            # Tomar muestra aleatoria de tamaño CF
            sample_indices = rng.choice(pop_size, size=cf, replace=False)

            # Encontrar el más cercano según distancia
            closest_idx = sample_indices[0]
            min_dist = _distance(off_ind, new_genotypes[closest_idx])
            for s_idx in sample_indices[1:]:
                d = _distance(off_ind, new_genotypes[s_idx])
                if d < min_dist:
                    min_dist = d
                    closest_idx = s_idx

            # Reemplazar SOLO si el descendiente es mejor (AMB-012, SUP-013)
            if off_apt > new_aptitude[closest_idx]:
                new_genotypes[closest_idx] = off_ind
                new_aptitude[closest_idx] = off_apt

        return Population(genotypes=new_genotypes, aptitude=new_aptitude)
