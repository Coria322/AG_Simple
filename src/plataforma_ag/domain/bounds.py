"""Validación y gestión de dominios por dimensión (RF-05, RF-10, AMB-002, AMB-011)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from plataforma_ag.domain import DTYPE
from plataforma_ag.domain.config import (
    DEMO_BOUND_MAX,
    DEMO_BOUND_MIN,
    MAX_BITS,
    MIN_BITS,
    VariableBound,
    variable_name,
)
from plataforma_ag.domain.enums import ChromosomeType
from plataforma_ag.domain.errors import ValidationError

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def validate_bound(
    bound: VariableBound, chromosome_type: ChromosomeType | None = None
) -> list[str]:
    """Valida un dominio de variable. Devuelve una lista de errores (vacía si es válido)."""
    errors: list[str] = []
    if not np.isfinite(bound.min):
        errors.append(f"{bound.variable}: el límite inferior debe ser un número real finito.")
    if not np.isfinite(bound.max):
        errors.append(f"{bound.variable}: el límite superior debe ser un número real finito.")
    if np.isfinite(bound.min) and np.isfinite(bound.max) and bound.min >= bound.max:
        errors.append(
            f"{bound.variable}: dominio invertido o vacío ({bound.min} ≥ {bound.max}); "
            "se requiere min < max."
        )

    # Bits solo se validan para representaciones binarias / Gray
    needs_bits = chromosome_type in (ChromosomeType.BINARY, ChromosomeType.GRAY, None)
    if needs_bits and not (MIN_BITS <= bound.bits <= MAX_BITS):
        errors.append(
            f"{bound.variable}: bits fuera de rango ({bound.bits}); "
            f"debe estar entre {MIN_BITS} y {MAX_BITS}."
        )

    return errors


def validate_bounds_list(
    bounds: Sequence[VariableBound], chromosome_type: ChromosomeType | None = None
) -> None:
    """Valida la lista completa de dominios. Lanza ValidationError si hay algún error."""
    if not bounds:
        raise ValidationError("Debe haber al menos un dominio por dimensión (N ≥ 1).")

    all_errors: list[str] = []
    for b in bounds:
        all_errors.extend(validate_bound(b, chromosome_type))

    if all_errors:
        raise ValidationError("; ".join(all_errors))


def adjust_bounds(
    current_bounds: Sequence[VariableBound], new_dimensions: int
) -> tuple[VariableBound, ...]:
    """Ajusta los dominios a `new_dimensions` conservando los valores existentes (RF-05)."""
    if new_dimensions < 1:
        raise ValidationError("Las dimensiones deben ser un entero ≥ 1.")

    result: list[VariableBound] = []
    for i in range(new_dimensions):
        var_name = variable_name(i)
        if i < len(current_bounds):
            # Conservar valores previos asegurando el nombre de variable correcto
            prev = current_bounds[i]
            result.append(
                VariableBound(
                    variable=var_name,
                    min=prev.min,
                    max=prev.max,
                    bits=prev.bits,
                )
            )
        else:
            # Nueva dimensión con valores por defecto
            result.append(
                VariableBound(
                    variable=var_name,
                    min=DEMO_BOUND_MIN,
                    max=DEMO_BOUND_MAX,
                )
            )

    return tuple(result)


def bounds_to_arrays(
    bounds: Sequence[VariableBound],
) -> tuple[FloatArray, FloatArray, IntArray]:
    """Convierte una lista de VariableBound en arrays vectorizados (lower, upper, bits)."""
    validate_bounds_list(bounds)
    lower = np.array([b.min for b in bounds], dtype=DTYPE)
    upper = np.array([b.max for b in bounds], dtype=DTYPE)
    bits = np.array([b.bits for b in bounds], dtype=np.int64)
    return lower, upper, bits
