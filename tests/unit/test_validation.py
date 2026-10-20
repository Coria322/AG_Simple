"""Pruebas unitarias de reglas de validación integral (RF-10, TASK-036)."""

from __future__ import annotations

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
    SelectionMethod,
)
from plataforma_ag.domain.validation import (
    validate_experiment_config,
    validate_number_format,
    validate_single_field,
)


def _create_valid_config() -> ExperimentConfig:
    bounds = (
        VariableBound(variable="x1", min=-5.0, max=5.0, bits=16),
        VariableBound(variable="x2", min=-5.0, max=5.0, bits=16),
    )
    return ExperimentConfig(
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
        ),
        operators=OperatorConfig(
            selection_method=SelectionMethod.TOURNAMENT,
            selection_params={"k": 3},
            crossover_method=CrossoverMethod.SINGLE_POINT,
            mutation_method=MutationMethod.SIMPLE_FLIP,
        ),
        stopping=StoppingCriteria(max_generations=100),
    )


def test_configuracion_valida_completa() -> None:
    # Criterio de aceptación 2: Dado todos los campos válidos, is_valid es True
    cfg = _create_valid_config()
    res = validate_experiment_config(cfg)
    assert res.is_valid is True
    assert len(res.errors) == 0


def test_pc_invalido_fuera_de_rango() -> None:
    # Criterio de aceptación 1: Dado Pc=2, marca p_crossover como inválido
    cfg = _create_valid_config()
    bad_cfg = ExperimentConfig(
        problem=cfg.problem,
        genetic=GeneticParameters(
            population_size=cfg.genetic.population_size,
            chromosome_type=cfg.genetic.chromosome_type,
            p_crossover=2.0,  # Inválido
            p_mutation=cfg.genetic.p_mutation,
        ),
        operators=cfg.operators,
        stopping=cfg.stopping,
    )
    res = validate_experiment_config(bad_cfg)
    assert res.is_valid is False
    assert "p_crossover" in res.errors


def test_formato_coma_decimal_rechazado() -> None:
    # Caso borde: coma decimal ("0,5") es inválida
    err = validate_number_format("0,5")
    assert err is not None
    assert "coma" in err.lower()

    assert validate_number_format("0.5") is None
    assert validate_number_format("  -12.34  ") is None


def test_formato_campo_vacio_rechazado() -> None:
    err = validate_number_format("   ")
    assert err is not None
    assert "vacío" in err.lower()


def test_dominio_invertido_detectado() -> None:
    cfg = _create_valid_config()
    inverted_bounds = (
        VariableBound(variable="x1", min=5.0, max=-5.0, bits=16),
        VariableBound(variable="x2", min=-5.0, max=5.0, bits=16),
    )
    bad_cfg = ExperimentConfig(
        problem=ProblemDefinition(
            expression=cfg.problem.expression,
            dimensions=2,
            bounds=inverted_bounds,
        ),
        genetic=cfg.genetic,
        operators=cfg.operators,
        stopping=cfg.stopping,
    )
    res = validate_experiment_config(bad_cfg)
    assert res.is_valid is False
    assert any("range" in k or "x1" in k for k in res.errors)


def test_operadores_incompatibles_detectados() -> None:
    # REAL con SINGLE_POINT
    cfg = _create_valid_config()
    bad_cfg = ExperimentConfig(
        problem=cfg.problem,
        genetic=GeneticParameters(
            population_size=50,
            chromosome_type=ChromosomeType.REAL,
            p_crossover=0.8,
            p_mutation=0.1,
        ),
        operators=OperatorConfig(
            selection_method=SelectionMethod.TOURNAMENT,
            crossover_method=CrossoverMethod.SINGLE_POINT,  # Solo para binario
            mutation_method=MutationMethod.UNIFORM,
        ),
        stopping=cfg.stopping,
    )
    res = validate_experiment_config(bad_cfg)
    assert res.is_valid is False
    assert "crossover_method" in res.errors


def test_rosenbrock_con_dimension_uno_detectado() -> None:
    cfg = _create_valid_config()
    bounds_1d = (VariableBound(variable="x1", min=-2.0, max=2.0),)
    bad_cfg = ExperimentConfig(
        problem=ProblemDefinition(
            expression="",
            dimensions=1,
            bounds=bounds_1d,
            benchmark=BenchmarkFunction.ROSENBROCK,
        ),
        genetic=cfg.genetic,
        operators=cfg.operators,
        stopping=cfg.stopping,
    )
    res = validate_experiment_config(bad_cfg)
    assert res.is_valid is False
    assert "dimensions" in res.errors


def test_validate_single_field_helper() -> None:
    assert validate_single_field("p_crossover", "1.5") is not None
    assert validate_single_field("p_crossover", "0.8") is None
    assert validate_single_field("population_size", "1") is not None
    assert validate_single_field("population_size", "50") is None
    assert validate_single_field("max_generations", "0") is not None
    assert validate_single_field("max_generations", "100") is None
