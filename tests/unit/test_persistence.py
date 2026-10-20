"""Pruebas unitarias de persistencia JSON y telemetría CSV (RF-22, RF-23, TASK-042, TASK-043)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plataforma_ag.domain.config import (
    ExperimentConfig,
    GeneticParameters,
    OperatorConfig,
    ProblemDefinition,
    StoppingCriteria,
    VariableBound,
)
from plataforma_ag.domain.enums import (
    AngleUnit,
    BenchmarkFunction,
    ChromosomeType,
    CrossoverMethod,
    FitnessTransform,
    MutationMethod,
    ObjectiveSense,
    RedundantSpaceStrategy,
    ReplacementMethod,
    SelectionMethod,
)
from plataforma_ag.domain.errors import IncompatibleOperatorError, PersistenceError
from plataforma_ag.domain.run import GenerationRecord, Individual
from plataforma_ag.services.config_store import (
    export_config_to_json,
    import_config_from_json,
)
from plataforma_ag.services.history import RunHistoryStore
from plataforma_ag.services.telemetry_csv import (
    export_run_to_csv,
    import_run_from_csv,
)


def _sample_config() -> ExperimentConfig:
    bounds = (
        VariableBound(variable="x1", min=-5.12, max=5.12, bits=16),
        VariableBound(variable="x2", min=-5.12, max=5.12, bits=16),
    )
    return ExperimentConfig(
        problem=ProblemDefinition(
            expression="x1^2 + x2^2",
            dimensions=2,
            bounds=bounds,
            benchmark=BenchmarkFunction.SPHERE,
            angle_unit=AngleUnit.RADIANS,
            objective=ObjectiveSense.MINIMIZE,
            redundant_space_strategy=RedundantSpaceStrategy.SATURATION,
            fitness_transform=FitnessTransform.NEGATE_PLUS_C,
        ),
        genetic=GeneticParameters(
            population_size=100,
            chromosome_type=ChromosomeType.BINARY,
            p_crossover=0.85,
            p_mutation=0.05,
            seed=42,
        ),
        operators=OperatorConfig(
            selection_method=SelectionMethod.TOURNAMENT,
            selection_params={"k": 3},
            crossover_method=CrossoverMethod.SINGLE_POINT,
            crossover_params={},
            mutation_method=MutationMethod.SIMPLE_FLIP,
            mutation_params={},
            replacement_method=ReplacementMethod.GENERATIONAL,
            elitism_count=2,
        ),
        stopping=StoppingCriteria(
            max_generations=150,
            stagnation_enabled=True,
            stagnation_generations=20,
            stagnation_epsilon=1e-5,
        ),
    )


def test_json_ida_y_vuelta_completa(tmp_path: Path) -> None:
    # Criterio 1: Exportar e importar deja todos los campos idénticos
    original_cfg = _sample_config()
    target_file = tmp_path / "config_test.json"

    export_config_to_json(original_cfg, target_file, project_name="Test Proyecto")
    assert target_file.is_file()

    loaded_cfg = import_config_from_json(target_file)

    assert loaded_cfg.problem.dimensions == original_cfg.problem.dimensions
    assert loaded_cfg.problem.expression == original_cfg.problem.expression
    assert loaded_cfg.problem.objective == original_cfg.problem.objective
    assert loaded_cfg.problem.bounds == original_cfg.problem.bounds
    assert loaded_cfg.genetic.population_size == original_cfg.genetic.population_size
    assert loaded_cfg.genetic.chromosome_type == original_cfg.genetic.chromosome_type
    assert loaded_cfg.genetic.p_crossover == original_cfg.genetic.p_crossover
    assert loaded_cfg.genetic.p_mutation == original_cfg.genetic.p_mutation
    assert loaded_cfg.genetic.seed == original_cfg.genetic.seed
    assert loaded_cfg.operators.selection_method == original_cfg.operators.selection_method
    assert loaded_cfg.operators.selection_params == original_cfg.operators.selection_params
    assert loaded_cfg.operators.elitism_count == original_cfg.operators.elitism_count
    assert loaded_cfg.stopping.max_generations == original_cfg.stopping.max_generations
    assert loaded_cfg.stopping.stagnation_enabled == original_cfg.stopping.stagnation_enabled


def test_json_rechaza_operador_incompatible(tmp_path: Path) -> None:
    # Criterio de aceptación 2: Dado REAL con SINGLE_POINT en el JSON, se rechaza
    target_file = tmp_path / "invalid_config.json"
    bad_data = {
        "problem_configuration": {
            "benchmark": "CUSTOM_EXPRESSION",
            "dimensions": 2,
            "expression": "x1 + x2",
            "bounds": [
                {"variable": "x1", "min": -5.0, "max": 5.0},
                {"variable": "x2", "min": -5.0, "max": 5.0},
            ],
        },
        "genetic_parameters": {
            "population_size": 50,
            "chromosome_type": "REAL",
            "p_crossover": 0.8,
            "p_mutation": 0.1,
        },
        "operators": {
            "selection_method": "TOURNAMENT",
            "crossover_method": "SINGLE_POINT",  # Incompatible con REAL
            "mutation_method": "UNIFORM",
            "replacement_method": "GENERATIONAL",
        },
        "stopping_criteria": {
            "max_generations": 100,
        },
    }

    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(bad_data, f)

    with pytest.raises(IncompatibleOperatorError, match="incompatible con cromosoma"):
        import_config_from_json(target_file)


def test_csv_telemetria_ida_y_vuelta_completa(tmp_path: Path) -> None:
    # Criterio 1: Exportar e importar CSV coincide exactamente en tabla y series
    history = RunHistoryStore(sense=ObjectiveSense.MINIMIZE)
    ind0 = Individual(
        individual_id=0, genotype=(1, 0, 1), phenotype=(2.5, -1.0), fitness=4.2, is_elite=True
    )
    ind1 = Individual(
        individual_id=1, genotype=(0, 1, 0), phenotype=(0.0, 3.0), fitness=9.0, is_elite=False
    )
    rec0 = GenerationRecord(
        generation=0,
        best_fitness=4.2,
        mean_fitness=6.6,
        elite_fitness_cumulative=4.2,
        individuals=(ind0, ind1),
    )
    history.add_generation(rec0)

    ind2 = Individual(
        individual_id=0, genotype=(1, 1, 1), phenotype=(1.0, 1.0), fitness=2.0, is_elite=True
    )
    ind3 = Individual(
        individual_id=1, genotype=(1, 0, 0), phenotype=(2.0, 0.0), fitness=4.0, is_elite=False
    )
    rec1 = GenerationRecord(
        generation=1,
        best_fitness=2.0,
        mean_fitness=3.0,
        elite_fitness_cumulative=2.0,
        individuals=(ind2, ind3),
    )
    history.add_generation(rec1)

    csv_path = tmp_path / "telemetry_test.csv"
    export_run_to_csv(history, csv_path)
    assert csv_path.is_file()

    loaded_history = import_run_from_csv(csv_path, sense=ObjectiveSense.MINIMIZE)

    assert loaded_history.total_generations == 2
    assert loaded_history.available_generations == [0, 1]

    # Comparar individuos de generación 0
    page_g0, total_g0, _ = loaded_history.get_individuals_page(0)
    assert total_g0 == 2
    assert page_g0[0].individual_id == 0
    assert pytest.approx(page_g0[0].fitness) == 4.2
    assert page_g0[0].is_elite is True

    # Comparar series de convergencia
    s_orig = history.get_convergence_series()
    s_loaded = loaded_history.get_convergence_series()
    assert s_loaded.generations == s_orig.generations
    assert pytest.approx(s_loaded.best_fitness) == s_orig.best_fitness
    assert pytest.approx(s_loaded.mean_fitness) == s_orig.mean_fitness
    assert pytest.approx(s_loaded.cumulative_elite) == s_orig.cumulative_elite


def test_csv_sin_columna_fitness_rechazado(tmp_path: Path) -> None:
    # Criterio de aceptación 2: CSV sin columna fitness informa la columna faltante
    csv_path = tmp_path / "corrupt.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("generation,individual_id,genotype,phenotype,is_elite\n")
        f.write("0,0,0101,[1.0, 2.0],1\n")

    with pytest.raises(PersistenceError, match="Falta la columna obligatoria 'fitness'"):
        import_run_from_csv(csv_path)
