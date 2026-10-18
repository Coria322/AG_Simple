"""TASK-007 — Pruebas de validación y gestión de dominios por dimensión (RF-05)."""

from __future__ import annotations

import pytest

from plataforma_ag.domain.bounds import (
    adjust_bounds,
    bounds_to_arrays,
    validate_bound,
    validate_bounds_list,
)
from plataforma_ag.domain.config import VariableBound
from plataforma_ag.domain.enums import ChromosomeType
from plataforma_ag.domain.errors import ValidationError


def test_dominio_invertido_detectado() -> None:
    # RF-05 Escenario 2: min=3, max=1
    bound = VariableBound(variable="x1", min=3.0, max=1.0, bits=16)
    errors = validate_bound(bound)
    assert len(errors) == 1
    assert "dominio invertido" in errors[0]

    with pytest.raises(ValidationError, match="dominio invertido"):
        validate_bounds_list([bound])


def test_dominio_igual_detectado() -> None:
    bound = VariableBound(variable="x1", min=2.0, max=2.0, bits=16)
    errors = validate_bound(bound)
    assert len(errors) == 1
    assert "min < max" in errors[0]


def test_bits_fuera_de_rango() -> None:
    bound_low = VariableBound(variable="x1", min=0.0, max=1.0, bits=0)
    bound_high = VariableBound(variable="x1", min=0.0, max=1.0, bits=65)

    assert len(validate_bound(bound_low)) == 1
    assert len(validate_bound(bound_high)) == 1

    # Para representación real los bits se ignoran y no dan error
    assert validate_bound(bound_low, ChromosomeType.REAL) == []


def test_ajustar_bounds_de_2_a_5_conserva_valores_existentes() -> None:
    # RF-05 Escenario 1: N cambia de 2 a 5
    initial = (
        VariableBound(variable="x1", min=-2.0, max=2.0, bits=8),
        VariableBound(variable="x2", min=0.0, max=10.0, bits=12),
    )

    adjusted = adjust_bounds(initial, new_dimensions=5)
    assert len(adjusted) == 5

    # x1 y x2 conservan exactamente sus valores
    assert adjusted[0].variable == "x1"
    assert adjusted[0].min == -2.0
    assert adjusted[0].max == 2.0
    assert adjusted[0].bits == 8

    assert adjusted[1].variable == "x2"
    assert adjusted[1].min == 0.0
    assert adjusted[1].max == 10.0
    assert adjusted[1].bits == 12

    # x3, x4, x5 creados con nombres correctos y valores por defecto
    assert adjusted[2].variable == "x3"
    assert adjusted[3].variable == "x4"
    assert adjusted[4].variable == "x5"


def test_ajustar_bounds_reducir_dimensiones() -> None:
    initial = (
        VariableBound(variable="x1", min=-2.0, max=2.0),
        VariableBound(variable="x2", min=0.0, max=10.0),
        VariableBound(variable="x3", min=-5.0, max=5.0),
    )
    adjusted = adjust_bounds(initial, new_dimensions=1)
    assert len(adjusted) == 1
    assert adjusted[0].variable == "x1"
    assert adjusted[0].min == -2.0


def test_bounds_to_arrays() -> None:
    bounds = [
        VariableBound(variable="x1", min=-5.0, max=5.0, bits=10),
        VariableBound(variable="x2", min=0.0, max=100.0, bits=20),
    ]
    lower, upper, bits = bounds_to_arrays(bounds)
    assert list(lower) == [-5.0, 0.0]
    assert list(upper) == [5.0, 100.0]
    assert list(bits) == [10, 20]
