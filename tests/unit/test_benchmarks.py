"""Pruebas unitarias del catálogo de benchmarks vectorizados (RF-01, TASK-035)."""

from __future__ import annotations

import numpy as np
import pytest

from plataforma_ag.core.problem.benchmarks import (
    BENCHMARK_CATALOG,
    BenchmarkEvaluator,
    get_benchmark,
)
from plataforma_ag.core.problem.objective import ObjectiveFunction
from plataforma_ag.domain.enums import BenchmarkFunction, ObjectiveSense
from plataforma_ag.domain.errors import ValidationError


def test_rastrigin_evaluacion_en_origen_devuelve_cero() -> None:
    # Criterio de aceptación 1: Dado Rastrigin N=2 en (0,0), f = 0.0
    spec = get_benchmark(BenchmarkFunction.RASTRIGIN)
    res = spec.evaluate(np.array([[0.0, 0.0]]))
    assert res.shape == (1,)
    assert pytest.approx(float(res[0]), abs=1e-12) == 0.0


def test_rosenbrock_requiere_minimo_dos_dimensiones() -> None:
    # Criterio de aceptación 2: Dado Rosenbrock N=1, se rechaza
    spec = get_benchmark(BenchmarkFunction.ROSENBROCK)
    with pytest.raises(ValidationError, match="requiere al menos 2 dimensiones"):
        spec.evaluate(np.array([[1.0]]))

    with pytest.raises(ValidationError, match="requiere al menos 2 dimensiones"):
        BenchmarkEvaluator(spec, dimensions=1)


def test_rosenbrock_evaluacion_en_optimo_devuelve_cero() -> None:
    spec = get_benchmark(BenchmarkFunction.ROSENBROCK)
    opt = spec.global_optimum_location(3)
    res = spec.evaluate(opt)
    assert pytest.approx(float(res[0]), abs=1e-12) == 0.0


def test_ackley_evaluacion_en_origen_con_tolerancia_estricta() -> None:
    # Caso borde: Ackley en el origen f(0..0) = 0 con tolerancia 1e-12
    spec = get_benchmark(BenchmarkFunction.ACKLEY)
    res = spec.evaluate(np.zeros((1, 5)))
    assert float(res[0]) == 0.0


def test_esfera_evaluacion_convexa() -> None:
    spec = get_benchmark(BenchmarkFunction.SPHERE)
    pts = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [-3.0, 4.0],
        ]
    )
    res = spec.evaluate(pts)
    assert pytest.approx(float(res[0]), abs=1e-12) == 0.0
    assert pytest.approx(float(res[1]), abs=1e-12) == 5.0
    assert pytest.approx(float(res[2]), abs=1e-12) == 25.0


def test_griewank_evaluacion_en_origen() -> None:
    # Griewank usa sqrt(i) con i desde 1
    spec = get_benchmark(BenchmarkFunction.GRIEWANK)
    res = spec.evaluate(np.zeros((2, 10)))
    assert pytest.approx(float(res[0]), abs=1e-12) == 0.0
    assert pytest.approx(float(res[1]), abs=1e-12) == 0.0


def test_catalogo_completo_disponible() -> None:
    assert len(BENCHMARK_CATALOG) == 5
    for fn in (
        BenchmarkFunction.SPHERE,
        BenchmarkFunction.RASTRIGIN,
        BenchmarkFunction.ROSENBROCK,
        BenchmarkFunction.ACKLEY,
        BenchmarkFunction.GRIEWANK,
    ):
        spec = get_benchmark(fn)
        assert spec.min_dimensions >= 1
        assert spec.default_bounds[0] < spec.default_bounds[1]
        assert spec.global_minimum_value == 0.0


def test_benchmark_evaluator_integracion_con_objective_function() -> None:
    spec = get_benchmark(BenchmarkFunction.SPHERE)
    evaluator = BenchmarkEvaluator(spec=spec, dimensions=2)
    obj = ObjectiveFunction(evaluator=evaluator, sense=ObjectiveSense.MINIMIZE)

    pop = np.array([[1.0, 2.0], [0.0, 0.0]])
    raw_fit, aptitude = obj.evaluate(pop)

    assert len(raw_fit) == 2
    assert pytest.approx(float(raw_fit[0])) == 5.0
    assert pytest.approx(float(raw_fit[1])) == 0.0
    # En minimización con negate_plus_c: el punto con raw=0 tiene mayor aptitud que raw=5
    assert aptitude[1] > aptitude[0]
