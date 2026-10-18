"""TASK-012 — Pruebas de criterios de parada (RF-09, AMB-009)."""

from __future__ import annotations

from plataforma_ag.core.engine.stopping import StoppingEvaluator
from plataforma_ag.domain.config import StoppingCriteria
from plataforma_ag.domain.enums import StopReason


def test_parada_por_max_generaciones() -> None:
    # RF-09 Escenario 1: Gmax=50 y estancamiento off -> termina en gen 50 con MAX_GENERATIONS
    criteria = StoppingCriteria(max_generations=50, stagnation_enabled=False)
    evaluator = StoppingEvaluator(criteria)

    stop, reasons = evaluator.evaluate(generation=49, best_fitness_history=[1.0] * 50)
    assert not stop
    assert reasons == ()

    stop, reasons = evaluator.evaluate(generation=50, best_fitness_history=[1.0] * 51)
    assert stop
    assert reasons == (StopReason.MAX_GENERATIONS,)


def test_parada_por_estancamiento() -> None:
    # RF-09 Escenario 2: K=25, eps=1e-6 y fitness constante 25 generaciones -> STAGNATION
    criteria = StoppingCriteria(
        max_generations=1000,
        stagnation_enabled=True,
        stagnation_generations=25,
        stagnation_epsilon=1e-6,
    )
    evaluator = StoppingEvaluator(criteria)

    # Menos de 25 generaciones: no se evalúa aún
    history = [10.0 - i * 0.1 for i in range(20)]
    stop, reasons = evaluator.evaluate(generation=19, best_fitness_history=history)
    assert not stop

    # 30 generaciones donde las últimas 26 tienen exactamente el mismo fitness
    history_stagnant = [5.0] * 30
    stop, reasons = evaluator.evaluate(generation=29, best_fitness_history=history_stagnant)
    assert stop
    assert reasons == (StopReason.STAGNATION,)


def test_ambos_criterios_disparados_simultaneamente() -> None:
    # Si coinciden en la misma generación, se registran ambos (RF-09 Casos borde)
    criteria = StoppingCriteria(
        max_generations=50,
        stagnation_enabled=True,
        stagnation_generations=10,
        stagnation_epsilon=1e-4,
    )
    evaluator = StoppingEvaluator(criteria)

    history = [2.0] * 60
    stop, reasons = evaluator.evaluate(generation=50, best_fitness_history=history)
    assert stop
    assert StopReason.MAX_GENERATIONS in reasons
    assert StopReason.STAGNATION in reasons
