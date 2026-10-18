"""TASK-006 — Pruebas de la función objetivo y transformaciones de aptitud (RF-04, AMB-001)."""

from __future__ import annotations

import numpy as np
import pytest

from plataforma_ag.core.problem.expression import compile_expression
from plataforma_ag.core.problem.objective import ObjectiveFunction
from plataforma_ag.domain.enums import FitnessTransform, ObjectiveSense


def test_esfera_2d_minimizar_mejor_individuo_menor_fx() -> None:
    # RF-04 Escenario 1: Minimizar
    evaluator = compile_expression("x1^2 + x2^2", dimensions=2)
    obj = ObjectiveFunction(
        evaluator=evaluator,
        sense=ObjectiveSense.MINIMIZE,
        transform=FitnessTransform.NEGATE_PLUS_C,
    )

    pop = np.array([
        [0.0, 0.0],   # f = 0.0 (mejor)
        [1.0, 1.0],   # f = 2.0
        [3.0, 4.0],   # f = 25.0 (peor)
    ])
    raw, aptitude = obj.evaluate(pop)

    assert raw[0] == 0.0
    assert raw[1] == 2.0
    assert raw[2] == 25.0

    # En minimización con NEGATE_PLUS_C, C = 25.0
    # aptitude = [25.0 - 0 = 25.0, 25.0 - 2 = 23.0, 25.0 - 25 = 0.0]
    assert aptitude[0] > aptitude[1] > aptitude[2]
    assert aptitude[0] == 25.0
    assert aptitude[2] == 0.0

    # Ordenar índices: el mejor debe ser el índice 0
    ranked = obj.rank_indices(raw)
    assert list(ranked) == [0, 1, 2]
    assert obj.is_better(raw[0], raw[1])
    assert not obj.is_better(raw[1], raw[0])


def test_esfera_2d_maximizar_mejor_individuo_mayor_fx() -> None:
    # RF-04 Escenario 2: Maximizar
    evaluator = compile_expression("x1^2 + x2^2", dimensions=2)
    obj = ObjectiveFunction(
        evaluator=evaluator,
        sense=ObjectiveSense.MAXIMIZE,
        transform=FitnessTransform.NEGATE_PLUS_C,
    )

    pop = np.array([
        [0.0, 0.0],   # f = 0.0 (peor)
        [1.0, 1.0],   # f = 2.0
        [3.0, 4.0],   # f = 25.0 (mejor)
    ])
    raw, aptitude = obj.evaluate(pop)

    assert aptitude[2] > aptitude[1] > aptitude[0]
    ranked = obj.rank_indices(raw)
    assert list(ranked) == [2, 1, 0]
    assert obj.is_better(raw[2], raw[1])


def test_aptitud_uniforme_sin_division_por_cero() -> None:
    evaluator = compile_expression("x1 + x2", dimensions=2)
    obj = ObjectiveFunction(
        evaluator=evaluator,
        sense=ObjectiveSense.MINIMIZE,
        transform=FitnessTransform.NEGATE_PLUS_C,
    )

    pop = np.array([[1.0, 1.0], [2.0, 0.0], [0.0, 2.0]])  # todos f = 2.0
    raw, aptitude = obj.evaluate(pop)
    assert np.all(raw == 2.0)
    # Todos deben tener aptitud positiva uniforme
    assert np.all(aptitude == 1.0)


def test_inverse_shift_transform() -> None:
    evaluator = compile_expression("x1^2", dimensions=1)
    obj = ObjectiveFunction(
        evaluator=evaluator,
        sense=ObjectiveSense.MINIMIZE,
        transform=FitnessTransform.INVERSE_SHIFT,
    )

    pop = np.array([[0.0], [2.0]])  # f = 0.0 (f_min), 4.0
    _raw, aptitude = obj.evaluate(pop)

    # 1 / (1 + 0 - 0) = 1.0
    assert aptitude[0] == 1.0
    # 1 / (1 + 4 - 0) = 0.2
    assert aptitude[1] == pytest.approx(0.2)
    assert aptitude[0] > aptitude[1]


def test_penalizados_reciben_aptitud_cero() -> None:
    evaluator = compile_expression("1/x1", dimensions=1)
    obj = ObjectiveFunction(
        evaluator=evaluator,
        sense=ObjectiveSense.MINIMIZE,
        transform=FitnessTransform.NEGATE_PLUS_C,
    )

    pop = np.array([[1.0], [0.0]])  # 1.0 finito, inf no finito
    raw, aptitude = obj.evaluate(pop)

    assert np.isfinite(raw[0])
    assert not np.isfinite(raw[1])
    assert aptitude[1] == 0.0
    assert obj.is_better(raw[0], raw[1])
    assert not obj.is_better(raw[1], raw[0])

    ranked = obj.rank_indices(raw)
    assert list(ranked) == [0, 1]
