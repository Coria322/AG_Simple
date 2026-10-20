"""Pruebas unitarias de los controladores MVC (Sprint 6)."""

from __future__ import annotations

from plataforma_ag.controllers.params_controller import ParamsController
from plataforma_ag.controllers.problem_controller import ProblemController
from plataforma_ag.controllers.run_controller import RunController
from plataforma_ag.controllers.telemetry_controller import TelemetryController
from plataforma_ag.domain.config import (
    ExperimentConfig,
    GeneticParameters,
    OperatorConfig,
    ProblemDefinition,
    StoppingCriteria,
    VariableBound,
)
from plataforma_ag.domain.enums import (
    BenchmarkFunction,
    ChromosomeType,
    CrossoverMethod,
    MutationMethod,
    ObjectiveSense,
    RunState,
)
from plataforma_ag.domain.events import GenerationCompleted
from plataforma_ag.domain.run import GenerationRecord, Individual


def test_problem_controller_benchmarks_y_dimensiones() -> None:
    ctrl = ProblemController()
    assert ctrl.problem.benchmark is BenchmarkFunction.CUSTOM_EXPRESSION

    # Seleccionar Sphere
    ctrl.set_benchmark(BenchmarkFunction.SPHERE)
    assert ctrl.problem.benchmark is BenchmarkFunction.SPHERE
    assert len(ctrl.problem.bounds) == 2

    # Cambiar dimensiones a 5
    ctrl.set_dimensions(5)
    assert ctrl.problem.dimensions == 5
    assert len(ctrl.problem.bounds) == 5
    assert ctrl.problem.bounds[4].variable == "x5"

    # Reducir a 3
    ctrl.set_dimensions(3)
    assert ctrl.problem.dimensions == 3
    assert len(ctrl.problem.bounds) == 3

    # Cambiar sentido
    ctrl.set_objective_sense(ObjectiveSense.MAXIMIZE)
    assert ctrl.problem.objective is ObjectiveSense.MAXIMIZE


def test_problem_controller_validacion_expresion() -> None:
    ctrl = ProblemController()
    assert ctrl.validate_expression("x1^2 + x2^2") is None
    # Variable fuera de rango (N=2 pero usa x3)
    assert ctrl.validate_expression("x1 + x3") is not None
    # Expresión con sintaxis errónea
    assert ctrl.validate_expression("x1 ++ /") is not None


def test_params_controller_compatibilidad_al_cambiar_cromosoma() -> None:
    ctrl = ParamsController()
    assert ctrl.genetic.chromosome_type is ChromosomeType.BINARY

    # Cambiar a PERMUTATION
    notifications: list[tuple[str, str]] = []
    ctrl.add_operator_notification_listener(lambda kind, op: notifications.append((kind, op)))

    ctrl.set_chromosome_type(ChromosomeType.PERMUTATION)
    assert ctrl.genetic.chromosome_type is ChromosomeType.PERMUTATION

    # El operador de cruce no puede ser SINGLE_POINT para permutaciones
    assert ctrl.operators.crossover_method in (
        CrossoverMethod.PMX,
        CrossoverMethod.OX,
        CrossoverMethod.CX,
    )
    # Mutación debe ser SWAP o INVERSION
    assert ctrl.operators.mutation_method in (
        MutationMethod.SWAP,
        MutationMethod.INVERSION,
    )
    assert len(notifications) > 0


def test_telemetry_controller_eventos_y_paginacion() -> None:
    ctrl = TelemetryController()
    ind1 = Individual(individual_id=0, genotype=(1, 0), phenotype=(1.0,), fitness=10.0)
    ind2 = Individual(individual_id=1, genotype=(0, 1), phenotype=(2.0,), fitness=5.0)

    rec = GenerationRecord(
        generation=1,
        best_fitness=5.0,
        mean_fitness=7.5,
        elite_fitness_cumulative=5.0,
        individuals=(ind1, ind2),
    )

    ctrl.handle_event(GenerationCompleted(generation=1, snapshot=rec))

    # Paginación
    page_items, total = ctrl.get_population_page(
        generation=1, page=1, page_size=10, sort_by="fitness"
    )
    assert total == 2
    assert len(page_items) == 2
    # Con sort_by fitness ascendente, 5.0 debe ir primero
    assert page_items[0].fitness == 5.0
    assert page_items[1].fitness == 10.0


def test_run_controller_ciclo_y_polling() -> None:
    telemetry = TelemetryController()
    run_ctrl = RunController(telemetry)

    assert run_ctrl.state is RunState.IDLE

    # Iniciar con configuración válida mínima
    cfg = ExperimentConfig(
        problem=ProblemDefinition(
            expression="x1^2",
            dimensions=1,
            bounds=(VariableBound(variable="x1", min=-2.0, max=2.0),),
            objective=ObjectiveSense.MINIMIZE,
        ),
        genetic=GeneticParameters(population_size=10, p_crossover=0.8, p_mutation=0.1),
        operators=OperatorConfig(),
        stopping=StoppingCriteria(max_generations=5),
    )

    run_ctrl.start_run(cfg)
    assert run_ctrl.state is RunState.RUNNING

    # Esperar y procesar eventos
    run_ctrl._core_controller.wait_until_finished(timeout=2.0)
    polled = run_ctrl.poll_events()
    assert polled > 0
    assert run_ctrl.state in (RunState.COMPLETED, RunState.RUNNING)
