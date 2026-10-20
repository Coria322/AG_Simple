"""Operadores de mutación genética (RF-15, AMB-004).

Implementa:
  - SIMPLE_FLIP: Flip bit con probabilidad Pm (Binario/Gray).
  - UNIFORM: Mutación uniforme en bits o en reales (Binario/Gray y Real).
  - NON_UNIFORM: Mutación no uniforme dependiente del tiempo t y Gmax (Real y Binario/Gray).
  - SWAP: Intercambio recíproco de dos posiciones (Permutaciones).
  - INVERSION: Inversión de un subsegmento continuo (Permutaciones).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from plataforma_ag.core.operators.base import Genotype, MutationContext, ParamSpec
from plataforma_ag.core.operators.registry import register_operator
from plataforma_ag.domain.enums import ChromosomeType, MutationMethod, OperatorKind

BINARY_AND_GRAY = frozenset({ChromosomeType.BINARY, ChromosomeType.GRAY})
PERMUTATION_ONLY = frozenset({ChromosomeType.PERMUTATION})
ALL_EXCEPT_PERM = frozenset(
    {
        ChromosomeType.BINARY,
        ChromosomeType.GRAY,
        ChromosomeType.REAL,
    }
)


@register_operator
class SimpleFlipMutation:
    """Mutación flip bit para cromosomas binarios/Gray."""

    name = MutationMethod.SIMPLE_FLIP.value
    kind = OperatorKind.MUTATION
    compatible_with = BINARY_AND_GRAY
    params_schema = ()

    def mutate(
        self, ind: Genotype, ctx: MutationContext, rng: np.random.Generator, **params: Any
    ) -> Genotype:
        pm = ctx.p_mutation
        if pm <= 0.0:
            return ind.copy()
        mutated = ind.copy()
        mask = rng.random(size=len(ind)) < pm
        mutated[mask] = 1 - mutated[mask]
        return mutated


@register_operator
class UniformMutation:
    """Mutación uniforme para cromosomas binarios, Gray y reales (AMB-004)."""

    name = MutationMethod.UNIFORM.value
    kind = OperatorKind.MUTATION
    compatible_with = ALL_EXCEPT_PERM
    params_schema = ()

    def mutate(
        self, ind: Genotype, ctx: MutationContext, rng: np.random.Generator, **params: Any
    ) -> Genotype:
        pm = ctx.p_mutation
        if pm <= 0.0:
            return ind.copy()

        mutated = ind.copy()
        mask = rng.random(size=len(ind)) < pm

        # Si el array es de tipo entero/bits
        if np.issubdtype(ind.dtype, np.integer):
            mutated[mask] = 1 - mutated[mask]
        else:  # Reales: reemplazar por valor uniforme en [lower, upper]
            dim_lower = ctx.lower
            dim_upper = ctx.upper
            replacements = rng.uniform(dim_lower, dim_upper)
            mutated[mask] = replacements[mask]

        return mutated


@register_operator
class NonUniformMutation:
    """Mutación no uniforme dinámica con fórmula Delta(t, y) = y * (1 - r^((1 - t/Gmax)^b))."""

    name = MutationMethod.NON_UNIFORM.value
    kind = OperatorKind.MUTATION
    compatible_with = ALL_EXCEPT_PERM
    params_schema = (
        ParamSpec(
            "b",
            float,
            2.0,
            label="Parámetro de forma (b)",
            minimum=0.0,
        ),
    )

    def mutate(
        self, ind: Genotype, ctx: MutationContext, rng: np.random.Generator, **params: Any
    ) -> Genotype:
        pm = ctx.p_mutation
        b = float(params.get("b", 2.0))
        t = ctx.generation
        g_max = max(ctx.max_generations, 1)

        # Fracción temporal acotada en [0, 1]
        tau = min(max(t / g_max, 0.0), 1.0)
        decay = (1.0 - tau) ** b

        mutated = ind.copy()

        # En cromosomas de bits (AMB-004): Pm efectiva decreciente
        if np.issubdtype(ind.dtype, np.integer):
            effective_pm = pm * decay
            if effective_pm <= 0.0:
                return mutated
            mask = rng.random(size=len(ind)) < effective_pm
            mutated[mask] = 1 - mutated[mask]
            return mutated

        # En cromosomas reales: perturbación decreciente Delta(t, y)
        mask = rng.random(size=len(ind)) < pm
        if not np.any(mask):
            return mutated

        for i in range(len(ind)):
            if mask[i]:
                x_val = mutated[i]
                low = ctx.lower[i]
                high = ctx.upper[i]

                # Dirección aleatoria: sumar o restar
                r = rng.random()
                delta_factor = 1.0 - (r**decay)

                if rng.random() < 0.5:
                    y = high - x_val
                    mutated[i] = x_val + y * delta_factor
                else:
                    y = x_val - low
                    mutated[i] = x_val - y * delta_factor

        # Clip de salvaguarda al dominio
        return np.clip(mutated, ctx.lower, ctx.upper)


@register_operator
class SwapMutation:
    """Mutación por intercambio recíproco para permutaciones."""

    name = MutationMethod.SWAP.value
    kind = OperatorKind.MUTATION
    compatible_with = PERMUTATION_ONLY
    params_schema = ()

    def mutate(
        self, ind: Genotype, ctx: MutationContext, rng: np.random.Generator, **params: Any
    ) -> Genotype:
        if rng.random() >= ctx.p_mutation or len(ind) < 2:
            return ind.copy()

        mutated = ind.copy()
        idx1, idx2 = rng.choice(len(ind), size=2, replace=False)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        return mutated


@register_operator
class InversionMutation:
    """Mutación por inversión de un segmento para permutaciones."""

    name = MutationMethod.INVERSION.value
    kind = OperatorKind.MUTATION
    compatible_with = PERMUTATION_ONLY
    params_schema = ()

    def mutate(
        self, ind: Genotype, ctx: MutationContext, rng: np.random.Generator, **params: Any
    ) -> Genotype:
        if rng.random() >= ctx.p_mutation or len(ind) < 2:
            return ind.copy()

        mutated = ind.copy()
        idx1, idx2 = sorted(rng.choice(len(ind), size=2, replace=False))
        mutated[idx1 : idx2 + 1] = mutated[idx1 : idx2 + 1][::-1]
        return mutated
