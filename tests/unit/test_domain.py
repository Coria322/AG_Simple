"""TASK-002 — Modelos de dominio, enums, errores y eventos."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from plataforma_ag.domain import DTYPE, enums
from plataforma_ag.domain.config import ExperimentConfig, ProblemDefinition, VariableBound
from plataforma_ag.domain.errors import ExpressionError, InvalidStateTransition, ValidationError
from plataforma_ag.domain.events import (
    DomainEvent,
    GenerationCompleted,
    ProgressChanged,
    RunCompleted,
)
from plataforma_ag.domain.run import GenerationRecord, Individual, RunHistory

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "plataforma_ag_agentic_docs/plataforma_ag/schemas/DOMAIN_SCHEMA.json"
)


def test_dtype_es_float64() -> None:
    assert DTYPE is np.float64


def test_experiment_config_por_defecto_cumple_el_ers() -> None:
    cfg = ExperimentConfig()
    assert cfg.genetic.population_size == 100
    assert cfg.genetic.p_crossover == 0.85
    assert cfg.genetic.p_mutation == 0.05
    assert cfg.stopping.max_generations == 200
    assert cfg.problem.objective is enums.ObjectiveSense.MINIMIZE
    assert cfg.problem.angle_unit is enums.AngleUnit.RADIANS
    assert cfg.problem.fitness_transform is enums.FitnessTransform.NEGATE_PLUS_C
    assert cfg.timestamp.tzinfo is not None


def test_dominios_de_demostracion() -> None:
    bounds = ProblemDefinition().bounds
    assert [b.variable for b in bounds] == ["x1", "x2"]
    assert bounds[0] == VariableBound("x1", -5.12, 5.12, 16)


@pytest.mark.skipif(not SCHEMA_PATH.exists(), reason="DOMAIN_SCHEMA.json no disponible")
def test_enums_coinciden_con_domain_schema() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["enumerations"], "el esquema debe declarar enumeraciones"
    for spec in schema["enumerations"]:
        enum_cls = getattr(enums, spec["name"])
        assert [m.value for m in enum_cls] == spec["values"], spec["name"]


def test_enums_se_serializan_como_cadena() -> None:
    assert json.dumps({"m": enums.SelectionMethod.TOURNAMENT}) == '{"m": "TOURNAMENT"}'
    assert str(enums.RunState.PAUSED) == "PAUSED"


def test_config_es_inmutable_y_se_modifica_con_replace() -> None:
    cfg = ExperimentConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.genetic.population_size = 5  # type: ignore[misc]
    nuevo = dataclasses.replace(cfg.genetic, population_size=50)
    assert nuevo.population_size == 50 and cfg.genetic.population_size == 100


def test_parametros_de_operador_no_se_comparten_entre_instancias() -> None:
    a, b = ExperimentConfig(), ExperimentConfig()
    a.operators.selection_params["k"] = 3
    assert b.operators.selection_params == {}


def test_evento_de_dominio_es_inmutable() -> None:
    event = ProgressChanged(current=50, maximum=200, percent=25.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.percent = 30.0  # type: ignore[misc]
    assert isinstance(event, DomainEvent)
    assert event.occurred_at.tzinfo is not None


def test_eventos_requieren_argumentos_por_nombre() -> None:
    with pytest.raises(TypeError):
        ProgressChanged(50, 200, 25.0)  # type: ignore[misc]


def test_generation_completed_transporta_la_instantanea() -> None:
    ind = Individual(0, (1, 0, 1), (0.5,), fitness=0.25, aptitude=1.0, is_elite=True)
    record = GenerationRecord(3, 0.25, 0.5, 0.25, (ind,))
    event = GenerationCompleted(generation=3, snapshot=record)
    assert event.snapshot.individuals[0].aptitude == 1.0


def test_run_completed_admite_varias_razones() -> None:
    reasons = (enums.StopReason.MAX_GENERATIONS, enums.StopReason.STAGNATION)
    assert RunCompleted(generation=50, stop_reasons=reasons).stop_reasons == reasons


def test_run_history_por_defecto() -> None:
    a, b = RunHistory(), RunHistory()
    assert a.state is enums.RunState.IDLE and a.run_id != b.run_id
    a.generations.append(GenerationRecord(0, 1.0, 1.0, 1.0))
    assert b.generations == []


def test_errores_con_contexto() -> None:
    assert str(ExpressionError("mal", position=4)) == "mal (posición 4)"
    assert (
        str(ValidationError("fuera de rango", field="p_crossover")) == "p_crossover: fuera de rango"
    )
    err = InvalidStateTransition(enums.RunState.IDLE, "pausar")
    assert "IDLE" in str(err) and err.requested == "pausar"
