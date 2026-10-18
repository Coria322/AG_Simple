"""TASK-033 — Pruebas de control de flujo, concurrencia y máquina de estados (RF-17, RNF-01)."""

from __future__ import annotations

import time

import pytest

from plataforma_ag.core.engine.run_controller import RunController
from plataforma_ag.domain.config import (
    ExperimentConfig,
    GeneticParameters,
    OperatorConfig,
    ProblemDefinition,
    StoppingCriteria,
    VariableBound,
)
from plataforma_ag.domain.enums import (
    ChromosomeType,
    CrossoverMethod,
    MutationMethod,
    ObjectiveSense,
    RunState,
    SelectionMethod,
)
from plataforma_ag.domain.errors import InvalidStateTransition
from plataforma_ag.domain.events import (
    GenerationCompleted,
    ProgressChanged,
    RunCancelled,
    RunCompleted,
    RunPaused,
    RunStarted,
)


def _create_fast_config(generations: int = 20) -> ExperimentConfig:
    bounds = (
        VariableBound(variable="x1", min=-2.0, max=2.0, bits=8),
        VariableBound(variable="x2", min=-2.0, max=2.0, bits=8),
    )
    return ExperimentConfig(
        problem=ProblemDefinition(
            expression="x1^2 + x2^2",
            dimensions=2,
            bounds=bounds,
            objective=ObjectiveSense.MINIMIZE,
        ),
        genetic=GeneticParameters(
            population_size=20,
            chromosome_type=ChromosomeType.BINARY,
            p_crossover=0.8,
            p_mutation=0.05,
            seed=42,
        ),
        operators=OperatorConfig(
            selection_method=SelectionMethod.TOURNAMENT,
            selection_params={"k": 2},
            crossover_method=CrossoverMethod.SINGLE_POINT,
            mutation_method=MutationMethod.SIMPLE_FLIP,
        ),
        stopping=StoppingCriteria(max_generations=generations),
    )


def test_transiciones_validas_corrida_completa() -> None:
    # IDLE -> start() -> RUNNING -> COMPLETED
    ctrl = RunController()
    assert ctrl.state is RunState.IDLE

    config = _create_fast_config(generations=10)
    ctrl.start(config)
    assert ctrl.state is RunState.RUNNING

    # Esperar terminación
    finished = ctrl.wait_until_finished(timeout=5.0)
    assert finished

    # Drenar eventos
    events = []
    while not ctrl.event_queue.empty():
        ev = ctrl.event_queue.get()
        ctrl.process_event(ev)
        events.append(ev)

    assert ctrl.state is RunState.COMPLETED

    event_types = [type(e) for e in events]
    assert RunStarted in event_types
    assert GenerationCompleted in event_types
    assert ProgressChanged in event_types
    assert RunCompleted in event_types


def test_transiciones_invalidas_lanzan_error() -> None:
    ctrl = RunController()
    assert ctrl.state is RunState.IDLE

    # No se puede pausar desde IDLE
    with pytest.raises(InvalidStateTransition, match="pausar"):
        ctrl.pause()

    # No se puede reanudar desde IDLE
    with pytest.raises(InvalidStateTransition, match="reanudar"):
        ctrl.resume()

    # Iniciar
    ctrl.start(_create_fast_config(generations=20))
    # No se puede iniciar otra vez mientras corre
    with pytest.raises(InvalidStateTransition, match="iniciar"):
        ctrl.start(_create_fast_config(generations=20))

    ctrl.abort()
    ctrl.wait_until_finished()


def test_pausa_y_reanudacion_sin_saltos() -> None:
    # RUNNING -> pause() -> PAUSED -> resume() -> RUNNING
    ctrl = RunController()
    config = _create_fast_config(generations=150)
    ctrl.start(config)

    # Esperar al menos la generación 2 para pausar en plena corrida
    paused = False
    for _ in range(100):
        time.sleep(0.005)
        while not ctrl.event_queue.empty():
            ev = ctrl.event_queue.get()
            ctrl.process_event(ev)
            if isinstance(ev, GenerationCompleted) and ev.generation >= 2:
                ctrl.pause()
                paused = True
                break
        if paused:
            break

    assert ctrl.state is RunState.PAUSED

    # Drenar hasta recibir RunPaused (momento en que el hilo de trabajo se detiene efectivamente)
    received_run_paused = False
    for _ in range(100):
        while not ctrl.event_queue.empty():
            ev = ctrl.event_queue.get()
            ctrl.process_event(ev)
            if isinstance(ev, RunPaused):
                received_run_paused = True
        if received_run_paused:
            break
        time.sleep(0.005)

    assert received_run_paused, "El worker debió emitir RunPaused al pausar"
    assert ctrl.state is RunState.PAUSED

    # Mientras está en pausa, esperar y confirmar que NO se emite ninguna nueva generación
    time.sleep(0.05)
    gens_while_waiting = []
    while not ctrl.event_queue.empty():
        ev = ctrl.event_queue.get()
        ctrl.process_event(ev)
        if isinstance(ev, GenerationCompleted):
            gens_while_waiting.append(ev.generation)

    assert len(gens_while_waiting) == 0, "No debe emitirse generaciones en pausa"
    assert ctrl.state is RunState.PAUSED

    # Reanudar
    ctrl.resume()
    assert ctrl.state is RunState.RUNNING

    ctrl.wait_until_finished(timeout=5.0)
    while not ctrl.event_queue.empty():
        ev = ctrl.event_queue.get()
        ctrl.process_event(ev)

    assert ctrl.state is RunState.COMPLETED


def test_cancelacion_cooperativa_en_menos_de_un_segundo() -> None:
    # RNF-01 Criterio: cuando se pide cancelar, el hilo termina en <= 1 s
    ctrl = RunController()
    config = _create_fast_config(generations=1000)
    ctrl.start(config)

    time.sleep(0.05)
    start_time = time.perf_counter()
    ctrl.abort()
    assert ctrl.state is RunState.CANCELLED

    finished = ctrl.wait_until_finished(timeout=1.0)
    elapsed = time.perf_counter() - start_time

    assert finished, "El hilo debió terminar en <= 1 s tras abortar"
    assert elapsed <= 1.0, f"Tiempo de cancelación ({elapsed}s) superó 1.0s"

    # Verificar que el evento de cancelación llegó
    cancelled_event_found = False
    while not ctrl.event_queue.empty():
        ev = ctrl.event_queue.get()
        if isinstance(ev, RunCancelled):
            cancelled_event_found = True
    assert cancelled_event_found
