"""Pruebas unitarias de generación de reportes PDF (RF-24, TASK-044)."""

from __future__ import annotations

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
    BenchmarkFunction,
    ChromosomeType,
    CrossoverMethod,
    MutationMethod,
    ObjectiveSense,
    SelectionMethod,
)
from plataforma_ag.domain.errors import ValidationError
from plataforma_ag.domain.run import GenerationRecord, Individual
from plataforma_ag.services.history import RunHistoryStore
from plataforma_ag.services.pdf_report import PdfReportService


def _create_sample_run(n_gens: int = 10) -> tuple[ExperimentConfig, RunHistoryStore]:
    bounds = (
        VariableBound(variable="x1", min=-5.0, max=5.0, bits=16),
        VariableBound(variable="x2", min=-5.0, max=5.0, bits=16),
    )
    config = ExperimentConfig(
        problem=ProblemDefinition(
            expression="x1^2 + x2^2",
            dimensions=2,
            bounds=bounds,
            benchmark=BenchmarkFunction.SPHERE,
            objective=ObjectiveSense.MINIMIZE,
        ),
        genetic=GeneticParameters(
            population_size=20,
            chromosome_type=ChromosomeType.BINARY,
            p_crossover=0.85,
            p_mutation=0.05,
            seed=42,
        ),
        operators=OperatorConfig(
            selection_method=SelectionMethod.TOURNAMENT,
            selection_params={"k": 2},
            crossover_method=CrossoverMethod.SINGLE_POINT,
            mutation_method=MutationMethod.SIMPLE_FLIP,
        ),
        stopping=StoppingCriteria(max_generations=n_gens),
    )

    history = RunHistoryStore(sense=ObjectiveSense.MINIMIZE)
    for g in range(n_gens):
        inds = []
        for i in range(20):
            val = float(10.0 / (g + 1) + i * 0.2)
            inds.append(
                Individual(
                    individual_id=i,
                    genotype=(0, 1) * 8,
                    phenotype=(float(i) * 0.1, -float(g) * 0.1),
                    fitness=val,
                    is_elite=(i == 0),
                )
            )
        rec = GenerationRecord(
            generation=g,
            best_fitness=inds[0].fitness,
            mean_fitness=float(sum(x.fitness for x in inds) / len(inds)),
            elite_fitness_cumulative=inds[0].fitness,
            individuals=tuple(inds),
        )
        history.add_generation(rec)

    return config, history


def test_generacion_reporte_pdf_completo(tmp_path: Path) -> None:
    # Criterio de aceptación 1: Dado una corrida finalizada, genera PDF con las 4 secciones
    cfg, history = _create_sample_run(n_gens=15)
    pdf_path = tmp_path / "reporte_test.pdf"

    generated_path = PdfReportService.generate_report(cfg, history, pdf_path)

    assert generated_path.is_file()
    assert generated_path.stat().st_size > 5000  # Contiene texto, tablas e imagen 300 DPI

    # Verificar cabecera mágica de PDF
    with open(generated_path, "rb") as f:
        header = f.read(5)
        assert header == b"%PDF-"


def test_generacion_reporte_sin_datos_lanza_error() -> None:
    # Criterio de aceptación 2: Dado que no hay corrida cargada, se explica por qué
    cfg, _ = _create_sample_run(n_gens=1)
    empty_history = RunHistoryStore(sense=ObjectiveSense.MINIMIZE)

    with pytest.raises(ValidationError, match="No hay registros de generaciones"):
        PdfReportService.generate_report(cfg, empty_history, "output.pdf")
