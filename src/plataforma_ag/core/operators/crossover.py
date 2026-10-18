"""Operadores de cruce genético (RF-14, SUP-013, AMB-012).

Implementa:
  - SINGLE_POINT: Cruce de un punto (Binario/Gray).
  - UNIFORM: Cruce uniforme con máscara aleatoria p=0.5 (Binario/Gray y Real).
  - ARITHMETIC: Cruce aritmético convexo con parámetro a en [0,1] (Real).
  - FLAT: Cruce plano en [min(P1, P2), max(P1, P2)] (Real).
  - BLX_ALPHA: Blend crossover extendido por alpha >= 0 (Real).
  - PMX: Partially Mapped Crossover (Permutaciones).
  - OX: Order Crossover (Permutaciones).
  - CX: Cycle Crossover (Permutaciones).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from plataforma_ag.core.operators.base import Genotype, ParamSpec
from plataforma_ag.core.operators.registry import register_operator
from plataforma_ag.domain.enums import ChromosomeType, CrossoverMethod, OperatorKind

BINARY_AND_GRAY = frozenset({ChromosomeType.BINARY, ChromosomeType.GRAY})
REAL_ONLY = frozenset({ChromosomeType.REAL})
PERMUTATION_ONLY = frozenset({ChromosomeType.PERMUTATION})
BINARY_GRAY_REAL = frozenset({ChromosomeType.BINARY, ChromosomeType.GRAY, ChromosomeType.REAL})


# --- Cruces Binarios y Reales Generales ---------------------------------------


@register_operator
class SinglePointCrossover:
    """Cruce de un punto de corte."""

    name = CrossoverMethod.SINGLE_POINT.value
    kind = OperatorKind.CROSSOVER
    compatible_with = BINARY_AND_GRAY
    params_schema = ()

    def cross(
        self, p1: Genotype, p2: Genotype, rng: np.random.Generator, **params: Any
    ) -> tuple[Genotype, Genotype]:
        length = len(p1)
        if length <= 1:
            return p1.copy(), p2.copy()
        pt = int(rng.integers(1, length))
        c1 = np.concatenate([p1[:pt], p2[pt:]])
        c2 = np.concatenate([p2[:pt], p1[pt:]])
        return c1, c2


@register_operator
class UniformCrossover:
    """Cruce uniforme con probabilidad 0.5 de intercambio por gen."""

    name = CrossoverMethod.UNIFORM.value
    kind = OperatorKind.CROSSOVER
    compatible_with = BINARY_GRAY_REAL
    params_schema = ()

    def cross(
        self, p1: Genotype, p2: Genotype, rng: np.random.Generator, **params: Any
    ) -> tuple[Genotype, Genotype]:
        mask = rng.integers(0, 2, size=len(p1), dtype=bool)
        c1 = np.where(mask, p1, p2)
        c2 = np.where(mask, p2, p1)
        return c1, c2


# --- Cruces Reales -----------------------------------------------------------


@register_operator
class ArithmeticCrossover:
    """Cruce aritmético: C1 = a*P2 + (1-a)*P1, C2 = (1-a)*P2 + a*P1 (con a en [0,1])."""

    name = CrossoverMethod.ARITHMETIC.value
    kind = OperatorKind.CROSSOVER
    compatible_with = REAL_ONLY
    params_schema = (
        ParamSpec(
            "a",
            float,
            0.5,
            label="Factor de combinación (a)",
            minimum=0.0,
            maximum=1.0,
        ),
    )

    def cross(
        self, p1: Genotype, p2: Genotype, rng: np.random.Generator, **params: Any
    ) -> tuple[Genotype, Genotype]:
        a = float(params.get("a", 0.5))
        # Para P1=(0,0), P2=(10,10) y a=0.3: C1 = (1-a)*P2 + a*P1 = 7, C2 = a*P2 + (1-a)*P1 = 3
        c1 = a * p1 + (1.0 - a) * p2
        c2 = (1.0 - a) * p1 + a * p2
        return c1, c2


@register_operator
class FlatCrossover:
    """Cruce plano: muestreo uniforme en [min(P1, P2), max(P1, P2)] (SUP-013)."""

    name = CrossoverMethod.FLAT.value
    kind = OperatorKind.CROSSOVER
    compatible_with = REAL_ONLY
    params_schema = ()

    def cross(
        self, p1: Genotype, p2: Genotype, rng: np.random.Generator, **params: Any
    ) -> tuple[Genotype, Genotype]:
        low = np.minimum(p1, p2)
        high = np.maximum(p1, p2)
        # Si low == high, añadir epsilon mínimo para evitar rango vacío
        equal_mask = low == high
        high_safe = np.where(equal_mask, high + 1e-12, high)
        c1 = rng.uniform(low, high_safe)
        c2 = rng.uniform(low, high_safe)
        return c1, c2


@register_operator
class BLXAlphaCrossover:
    """Cruce BLX-alpha extendido por d_i * alpha."""

    name = CrossoverMethod.BLX_ALPHA.value
    kind = OperatorKind.CROSSOVER
    compatible_with = REAL_ONLY
    params_schema = (
        ParamSpec("alpha", float, 0.5, label="Parámetro alpha", minimum=0.0),
    )

    def cross(
        self, p1: Genotype, p2: Genotype, rng: np.random.Generator, **params: Any
    ) -> tuple[Genotype, Genotype]:
        alpha = float(params.get("alpha", 0.5))
        low = np.minimum(p1, p2)
        high = np.maximum(p1, p2)
        diff = high - low
        ext_low = low - alpha * diff
        ext_high = high + alpha * diff
        equal_mask = ext_low == ext_high
        ext_high_safe = np.where(equal_mask, ext_high + 1e-12, ext_high)
        c1 = rng.uniform(ext_low, ext_high_safe)
        c2 = rng.uniform(ext_low, ext_high_safe)
        return c1, c2


# --- Cruces de Permutaciones -------------------------------------------------


@register_operator
class PMXCrossover:
    """Partially Mapped Crossover (PMX) para permutaciones."""

    name = CrossoverMethod.PMX.value
    kind = OperatorKind.CROSSOVER
    compatible_with = PERMUTATION_ONLY
    params_schema = ()

    def cross(
        self, p1: Genotype, p2: Genotype, rng: np.random.Generator, **params: Any
    ) -> tuple[Genotype, Genotype]:
        n = len(p1)
        if n <= 1:
            return p1.copy(), p2.copy()

        cx1, cx2 = sorted(rng.choice(n, size=2, replace=False))
        cx2 += 1  # intervalo semiabierto [cx1, cx2)

        def pmx_single(parent1: Genotype, parent2: Genotype) -> Genotype:
            child = np.full(n, -1, dtype=parent1.dtype)
            # Copiar segmento
            child[cx1:cx2] = parent1[cx1:cx2]

            # Mapear elementos del segmento de parent2
            for i in range(cx1, cx2):
                val = parent2[i]
                if val not in child[cx1:cx2]:
                    # Buscar la posición equivalente
                    pos = i
                    while cx1 <= pos < cx2:
                        target = parent1[pos]
                        pos = int(np.where(parent2 == target)[0][0])
                    child[pos] = val

            # Llenar el resto directamente de parent2
            empty_mask = child == -1
            child[empty_mask] = parent2[empty_mask]
            return child

        return pmx_single(p1, p2), pmx_single(p2, p1)


@register_operator
class OXCrossover:
    """Order Crossover (OX) para permutaciones."""

    name = CrossoverMethod.OX.value
    kind = OperatorKind.CROSSOVER
    compatible_with = PERMUTATION_ONLY
    params_schema = ()

    def cross(
        self, p1: Genotype, p2: Genotype, rng: np.random.Generator, **params: Any
    ) -> tuple[Genotype, Genotype]:
        n = len(p1)
        if n <= 1:
            return p1.copy(), p2.copy()

        cx1, cx2 = sorted(rng.choice(n, size=2, replace=False))
        cx2 += 1

        def ox_single(parent1: Genotype, parent2: Genotype) -> Genotype:
            child = np.full(n, -1, dtype=parent1.dtype)
            child[cx1:cx2] = parent1[cx1:cx2]
            segment_set = set(parent1[cx1:cx2])

            # Recorrer parent2 circularmente desde cx2
            p2_order = [parent2[(cx2 + i) % n] for i in range(n)]
            remaining = [v for v in p2_order if v not in segment_set]

            idx = 0
            for i in range(n):
                pos = (cx2 + i) % n
                if child[pos] == -1:
                    child[pos] = remaining[idx]
                    idx += 1
            return child

        return ox_single(p1, p2), ox_single(p2, p1)


@register_operator
class CXCrossover:
    """Cycle Crossover (CX) para permutaciones."""

    name = CrossoverMethod.CX.value
    kind = OperatorKind.CROSSOVER
    compatible_with = PERMUTATION_ONLY
    params_schema = ()

    def cross(
        self, p1: Genotype, p2: Genotype, rng: np.random.Generator, **params: Any
    ) -> tuple[Genotype, Genotype]:
        n = len(p1)
        if n <= 1:
            return p1.copy(), p2.copy()

        def cx_single(parent1: Genotype, parent2: Genotype) -> Genotype:
            child = np.full(n, -1, dtype=parent1.dtype)
            visited = np.zeros(n, dtype=bool)

            cycle_idx = 0
            while not np.all(visited):
                start = int(np.where(~visited)[0][0])
                curr = start
                cycle = []

                while True:
                    visited[curr] = True
                    cycle.append(curr)
                    val2 = parent2[curr]
                    curr = int(np.where(parent1 == val2)[0][0])
                    if curr == start:
                        break

                # Ciclos pares toman de parent1, ciclos impares toman de parent2
                src = parent1 if (cycle_idx % 2 == 0) else parent2
                for pos in cycle:
                    child[pos] = src[pos]

                cycle_idx += 1

            return child

        return cx_single(p1, p2), cx_single(p2, p1)
