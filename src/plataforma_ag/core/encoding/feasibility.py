"""Mapeo fenotípico y políticas de factibilidad (RF-06, AMB-002).

Implementa la fórmula del ERS §2.3:
    x_i = x_min + D_i · (x_max - x_min) / (2^m_i - 1)
con soporte de precisión exacta para m hasta 64 bits y recorte final obligatorio (clip).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from plataforma_ag.domain import DTYPE
from plataforma_ag.domain.bounds import bounds_to_arrays
from plataforma_ag.domain.config import VariableBound
from plataforma_ag.domain.enums import RedundantSpaceStrategy

FloatArray = NDArray[np.float64]


def decode_binary_integer(bits: NDArray[np.uint8]) -> int:
    """Convierte un vector 1D de bits {0, 1} a su entero decimal correspondiente."""
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def decode_binary_dimension(
    d_ints: Sequence[int],
    m_bits: int,
    x_min: float,
    x_max: float,
) -> FloatArray:
    """Decodifica una lista/vector de enteros D a fenotipos continuos x_i en [x_min, x_max].

    Maneja división exacta con enteros arbitrarios de Python para evitar desbordes cuando m > 53.
    Aplica clip final obligatorio para garantizar x_min <= x <= x_max ante redondeos IEEE 754.
    """
    if m_bits < 1:
        raise ValueError(f"m_bits debe ser ≥ 1 (recibido {m_bits})")

    denom = (1 << m_bits) - 1
    rng_val = x_max - x_min

    results = np.empty(len(d_ints), dtype=DTYPE)
    for idx, d in enumerate(d_ints):
        if denom == 0:
            val = x_min
        else:
            # División precisa con enteros de Python
            ratio = float(d / denom)
            val = x_min + ratio * rng_val
        results[idx] = val

    # Clip final estricto al dominio [x_min, x_max] (RF-06)
    return np.clip(results, x_min, x_max)


def apply_feasibility_policy(
    phenotypes: FloatArray,
    bounds: Sequence[VariableBound],
    strategy: RedundantSpaceStrategy = RedundantSpaceStrategy.SATURATION,
) -> FloatArray:
    """Aplica la política de espacio redundante a una matriz de fenotipos (n, N)."""
    lower, upper, _ = bounds_to_arrays(bounds)
    if strategy is RedundantSpaceStrategy.SATURATION:
        return np.clip(phenotypes, lower, upper)

    # Para REJECTION_RESAMPLE: los valores fuera de rango son recortados como salvaguarda
    # (la regeneración activa se efectúa a nivel de codec / generador)
    return np.clip(phenotypes, lower, upper)
