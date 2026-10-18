"""TASK-030 — Pruebas de la matriz de habilitación y compatibilidad (RF-12)."""

from __future__ import annotations

import pytest

from plataforma_ag.core.operators.compatibility import (
    COMPATIBILITY_MATRIX,
    is_operator_allowed,
    validate_operator_config,
)
from plataforma_ag.domain.config import OperatorConfig
from plataforma_ag.domain.enums import (
    ChromosomeType,
    CrossoverMethod,
    MutationMethod,
    OperatorKind,
    SelectionMethod,
)
from plataforma_ag.domain.errors import IncompatibleOperatorError


def test_representacion_real_deshabilita_un_punto_y_flip_bit() -> None:
    # RF-12 Escenario 1: Representación Real
    assert not is_operator_allowed(
        ChromosomeType.REAL, OperatorKind.CROSSOVER, CrossoverMethod.SINGLE_POINT.value
    )
    assert not is_operator_allowed(
        ChromosomeType.REAL, OperatorKind.MUTATION, MutationMethod.SIMPLE_FLIP.value
    )

    # Pero sí permite Cruce Aritmético y Mutación Uniforme
    assert is_operator_allowed(
        ChromosomeType.REAL, OperatorKind.CROSSOVER, CrossoverMethod.ARITHMETIC.value
    )
    assert is_operator_allowed(
        ChromosomeType.REAL, OperatorKind.MUTATION, MutationMethod.UNIFORM.value
    )


def test_permutaciones_deshabilita_ranking_no_lineal() -> None:
    # Matriz RF-12: Ranking no lineal deshabilitado para permutaciones
    assert not is_operator_allowed(
        ChromosomeType.PERMUTATION,
        OperatorKind.SELECTION,
        SelectionMethod.NONLINEAR_RANKING.value,
    )
    # Permutaciones solo permiten PMX, OX, CX y Swap, Inversion
    rule = COMPATIBILITY_MATRIX[ChromosomeType.PERMUTATION]
    assert rule.allowed_crossover == frozenset({
        CrossoverMethod.PMX,
        CrossoverMethod.OX,
        CrossoverMethod.CX,
    })
    assert rule.allowed_mutation == frozenset({
        MutationMethod.SWAP,
        MutationMethod.INVERSION,
    })


def test_validar_config_incompatible_lanza_error() -> None:
    # Configuración inválida: REAL + SINGLE_POINT
    invalid_config = OperatorConfig(
        crossover_method=CrossoverMethod.SINGLE_POINT,
    )
    with pytest.raises(
        IncompatibleOperatorError, match="no es compatible con la representación REAL"
    ):
        validate_operator_config(ChromosomeType.REAL, invalid_config)


def test_validar_config_valida_no_lanza() -> None:
    valid_config = OperatorConfig(
        selection_method=SelectionMethod.TOURNAMENT,
        crossover_method=CrossoverMethod.ARITHMETIC,
        mutation_method=MutationMethod.NON_UNIFORM,
    )
    validate_operator_config(ChromosomeType.REAL, valid_config)
