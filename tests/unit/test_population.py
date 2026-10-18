"""TASK-010 — Pruebas de inicialización de población (RF-07)."""

from __future__ import annotations

import numpy as np
import pytest

from plataforma_ag.core.encoding.codecs import BinaryCodec, RealCodec
from plataforma_ag.core.engine.population import initialize_population
from plataforma_ag.domain.config import VariableBound
from plataforma_ag.domain.errors import ValidationError


def test_poblacion_por_defecto_tiene_100_individuos() -> None:
    # RF-07 Escenario 1
    bounds = (
        VariableBound(variable="x1", min=-5.0, max=5.0, bits=8),
        VariableBound(variable="x2", min=-5.0, max=5.0, bits=8),
    )
    codec = BinaryCodec(bounds)
    pop = initialize_population(codec)
    assert pop.shape[0] == 100
    assert pop.shape[1] == 16


def test_poblacion_invalida_menor_a_2_lanza_validation_error() -> None:
    # RF-07 Escenario 2
    bounds = (VariableBound(variable="x1", min=0.0, max=1.0),)
    codec = RealCodec(bounds)

    with pytest.raises(ValidationError, match="≥ 2"):
        initialize_population(codec, population_size=1)

    with pytest.raises(ValidationError, match="entero"):
        initialize_population(codec, population_size=True)  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="entero"):
        initialize_population(codec, population_size="100")  # type: ignore[arg-type]


def test_poblacion_con_semilla_reproducible() -> None:
    bounds = (VariableBound(variable="x1", min=0.0, max=1.0),)
    codec = RealCodec(bounds)

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    pop1 = initialize_population(codec, population_size=10, rng=rng1)
    pop2 = initialize_population(codec, population_size=10, rng=rng2)

    np.testing.assert_array_equal(pop1, pop2)
