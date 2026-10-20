"""TASK-019 — Pruebas del motor genético GAEngine (RF-07..RF-17)."""

from __future__ import annotations

import numpy as np

from plataforma_ag.core.engine.ga_engine import GAEngine
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
    SelectionMethod,
    StopReason,
)


def test_ga_engine_esfera_2d_converge_hacia_cero() -> None:
    # Optimización de la función Esfera 2D (x1^2 + x2^2) en [-5, 5]
    bounds = (
        VariableBound(variable="x1", min=-5.0, max=5.0, bits=16),
        VariableBound(variable="x2", min=-5.0, max=5.0, bits=16),
    )
    config = ExperimentConfig(
        problem=ProblemDefinition(
            expression="x1^2 + x2^2",
            dimensions=2,
            bounds=bounds,
            objective=ObjectiveSense.MINIMIZE,
        ),
        genetic=GeneticParameters(
            population_size=50,
            chromosome_type=ChromosomeType.BINARY,
            p_crossover=0.85,
            p_mutation=0.05,
            seed=42,
        ),
        operators=OperatorConfig(
            selection_method=SelectionMethod.TOURNAMENT,
            selection_params={"k": 3},
            crossover_method=CrossoverMethod.SINGLE_POINT,
            mutation_method=MutationMethod.SIMPLE_FLIP,
        ),
        stopping=StoppingCriteria(max_generations=25),
    )

    engine = GAEngine.from_config(config)
    records = []
    final_reasons = None

    for record, stop_reasons in engine.run():
        records.append(record)
        if stop_reasons is not None:
            final_reasons = stop_reasons

    # Debe haber ejecutado de la generación 0 a la 25 (26 generaciones)
    assert len(records) == 26
    assert records[0].generation == 0
    assert records[-1].generation == 25
    assert final_reasons == (StopReason.MAX_GENERATIONS,)

    # El mejor fitness final debe haber mejorado sustancialmente respecto al inicial
    f_initial = records[0].best_fitness
    f_final = records[-1].best_fitness
    assert f_final < f_initial, (
        f"Fitness final ({f_final}) debió ser menor que inicial ({f_initial})"
    )

    # Verificar estructura completa de Individuals
    gen0 = records[0]
    assert len(gen0.individuals) == 50
    elite_count = sum(1 for ind in gen0.individuals if ind.is_elite)
    assert elite_count == 1
    assert gen0.individuals[0].fitness is not None
    assert gen0.individuals[0].aptitude is not None


def test_ga_engine_cromosomas_reales_con_blx_alpha() -> None:
    # Optimización continua real
    bounds = (
        VariableBound(variable="x1", min=-5.12, max=5.12),
        VariableBound(variable="x2", min=-5.12, max=5.12),
    )
    config = ExperimentConfig(
        problem=ProblemDefinition(
            expression="x1^2 + x2^2",
            dimensions=2,
            bounds=bounds,
            objective=ObjectiveSense.MINIMIZE,
        ),
        genetic=GeneticParameters(
            population_size=30,
            chromosome_type=ChromosomeType.REAL,
            p_crossover=0.9,
            p_mutation=0.1,
            seed=123,
        ),
        operators=OperatorConfig(
            selection_method=SelectionMethod.TOURNAMENT,
            selection_params={"k": 2},
            crossover_method=CrossoverMethod.BLX_ALPHA,
            crossover_params={"alpha": 0.5},
            mutation_method=MutationMethod.NON_UNIFORM,
            mutation_params={"b": 2.0},
        ),
        stopping=StoppingCriteria(max_generations=15),
    )

    engine = GAEngine.from_config(config)
    last_record = None
    for record, _ in engine.run():
        last_record = record

    assert last_record is not None
    assert last_record.generation == 15
    assert np.isfinite(last_record.best_fitness)


def test_ga_engine_permutaciones_con_pmx_e_inversion() -> None:
    # Ejecución con cromosomas de permutación
    bounds = tuple(VariableBound(variable=f"x{i + 1}", min=0.0, max=10.0) for i in range(5))
    config = ExperimentConfig(
        problem=ProblemDefinition(
            expression="x1 + x2 + x3 + x4 + x5",
            dimensions=5,
            bounds=bounds,
            objective=ObjectiveSense.MINIMIZE,
        ),
        genetic=GeneticParameters(
            population_size=20,
            chromosome_type=ChromosomeType.PERMUTATION,
            p_crossover=0.8,
            p_mutation=0.2,
            seed=99,
        ),
        operators=OperatorConfig(
            selection_method=SelectionMethod.TOURNAMENT,
            selection_params={"k": 2},
            crossover_method=CrossoverMethod.PMX,
            mutation_method=MutationMethod.INVERSION,
        ),
        stopping=StoppingCriteria(max_generations=10),
    )

    engine = GAEngine.from_config(config)
    for record, _ in engine.run():
        # Verificar que cada individuo en la población sea una permutación válida 0..4
        for ind in record.individuals:
            assert sorted(ind.genotype) == [0, 1, 2, 3, 4]
