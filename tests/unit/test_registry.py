"""TASK-003 — Strategy base, OperatorRegistry y ParamSpec."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from plataforma_ag.core.operators.base import (
    ALL_CHROMOSOME_TYPES,
    CrossoverOperator,
    ParamSpec,
    SelectionOperator,
    resolve_params,
)
from plataforma_ag.core.operators.registry import OperatorRegistry
from plataforma_ag.domain.enums import ChromosomeType, OperatorKind
from plataforma_ag.domain.errors import (
    DuplicateOperatorError,
    UnknownOperatorError,
    ValidationError,
)

REAL_ONLY = frozenset({ChromosomeType.REAL})


class FakeTournament:
    name = "TOURNAMENT"
    kind = OperatorKind.SELECTION
    compatible_with = ALL_CHROMOSOME_TYPES
    params_schema = (ParamSpec("k", int, 3, label="Tamaño de torneo", minimum=2),)

    def select(self, fitness: Any, n: int, rng: np.random.Generator, **params: Any) -> Any:
        return np.zeros(n, dtype=np.intp)


class FakeArithmetic:
    name = "ARITHMETIC"
    kind = OperatorKind.CROSSOVER
    compatible_with = REAL_ONLY
    params_schema = (ParamSpec("a", float, 0.5, minimum=0.0, maximum=1.0),)

    def cross(self, p1: Any, p2: Any, rng: np.random.Generator, **params: Any) -> Any:
        a = params["a"]
        return a * p1 + (1 - a) * p2, (1 - a) * p1 + a * p2


class FakeUniformCrossover:
    name = "UNIFORM"
    kind = OperatorKind.CROSSOVER
    compatible_with = frozenset({ChromosomeType.BINARY, ChromosomeType.GRAY})
    params_schema = ()

    def cross(self, p1: Any, p2: Any, rng: np.random.Generator, **params: Any) -> Any:
        return p1, p2


class FakeUniformMutation:
    name = "UNIFORM"
    kind = OperatorKind.MUTATION
    compatible_with = ALL_CHROMOSOME_TYPES
    params_schema = ()

    def mutate(self, ind: Any, ctx: Any, rng: np.random.Generator, **params: Any) -> Any:
        return ind


@pytest.fixture
def registry() -> OperatorRegistry:
    return OperatorRegistry()


# --- Criterios de aceptación -------------------------------------------------


def test_nombre_repetido_lanza_duplicate_operator_error(registry: OperatorRegistry) -> None:
    registry.register(FakeTournament)
    with pytest.raises(DuplicateOperatorError, match="TOURNAMENT"):
        registry.register(FakeTournament())


def test_list_filtra_por_compatibilidad(registry: OperatorRegistry) -> None:
    registry.register(FakeArithmetic)
    registry.register(FakeUniformCrossover)
    assert registry.names(OperatorKind.CROSSOVER, ChromosomeType.REAL) == ["ARITHMETIC"]
    assert registry.names(OperatorKind.CROSSOVER, ChromosomeType.BINARY) == ["UNIFORM"]
    assert registry.names(OperatorKind.CROSSOVER, ChromosomeType.PERMUTATION) == []
    assert registry.names(OperatorKind.CROSSOVER) == ["ARITHMETIC", "UNIFORM"]


# --- Comportamiento adicional ------------------------------------------------


def test_mismo_nombre_en_distinta_familia_no_choca(registry: OperatorRegistry) -> None:
    registry.register(FakeUniformCrossover)
    registry.register(FakeUniformMutation)
    assert len(registry) == 2
    assert (OperatorKind.MUTATION, "UNIFORM") in registry


def test_register_funciona_como_decorador(registry: OperatorRegistry) -> None:
    @registry.register
    class DummyCrossover:
        name = "DUMMY"
        kind = OperatorKind.CROSSOVER
        compatible_with = REAL_ONLY
        params_schema = ()

        def cross(self, p1: Any, p2: Any, rng: Any, **params: Any) -> Any:
            return p2, p1

    assert isinstance(DummyCrossover, type)  # el decorador devuelve la clase intacta
    op = registry.get(OperatorKind.CROSSOVER, "DUMMY")
    assert isinstance(op, DummyCrossover)
    assert isinstance(op, CrossoverOperator)


def test_get_inexistente(registry: OperatorRegistry) -> None:
    with pytest.raises(UnknownOperatorError):
        registry.get(OperatorKind.SELECTION, "NO_EXISTE")


def test_unregister(registry: OperatorRegistry) -> None:
    registry.register(FakeTournament)
    registry.unregister(OperatorKind.SELECTION, "TOURNAMENT")
    assert len(registry) == 0
    with pytest.raises(UnknownOperatorError):
        registry.unregister(OperatorKind.SELECTION, "TOURNAMENT")


def test_contrato_incompleto_se_rechaza(registry: OperatorRegistry) -> None:
    class SinNombre:
        kind = OperatorKind.SELECTION
        compatible_with = ALL_CHROMOSOME_TYPES
        params_schema = ()

        def select(self, *a: Any) -> None: ...

    class SinMetodo:
        name = "X"
        kind = OperatorKind.MUTATION
        compatible_with = ALL_CHROMOSOME_TYPES
        params_schema = ()

    with pytest.raises(TypeError, match="name"):
        registry.register(SinNombre)
    with pytest.raises(TypeError, match="mutate"):
        registry.register(SinMetodo)


def test_protocols_runtime_checkable() -> None:
    assert isinstance(FakeTournament(), SelectionOperator)
    assert not isinstance(FakeArithmetic(), SelectionOperator)


def test_estrategia_se_ejecuta_con_parametros_resueltos() -> None:
    op = FakeArithmetic()
    params = resolve_params(op.params_schema, {"a": 0.3})
    c1, c2 = op.cross(
        np.array([0.0, 0.0]), np.array([10.0, 10.0]), np.random.default_rng(0), **params
    )
    np.testing.assert_allclose(c1, [7.0, 7.0])
    np.testing.assert_allclose(c2, [3.0, 3.0])


# --- ParamSpec / resolve_params ----------------------------------------------


def test_resolve_params_usa_defaults() -> None:
    assert resolve_params(FakeTournament.params_schema) == {"k": 3}


@pytest.mark.parametrize(
    ("spec", "value", "message"),
    [
        (ParamSpec("k", int, 3, minimum=2), 1, "≥ 2"),
        (ParamSpec("k", int, 3, minimum=2), 2.5, "entero"),
        (ParamSpec("k", int, 3, minimum=2), True, "numérico"),
        (
            ParamSpec(
                "q", float, 0.5, minimum=0, maximum=1, min_inclusive=False, max_inclusive=False
            ),
            1.0,
            "< 1",
        ),
        (ParamSpec("q", float, 0.5, minimum=0, maximum=1, min_inclusive=False), 0.0, "> 0"),
        (ParamSpec("a", float, 0.5), "abc", "formato"),
        (ParamSpec("a", float, 0.5), float("nan"), "finito"),
    ],
)
def test_param_spec_rechaza_valores_invalidos(spec: ParamSpec, value: Any, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        spec.validate(value)


def test_resolve_params_rechaza_desconocidos() -> None:
    with pytest.raises(ValidationError, match="desconocidos: z"):
        resolve_params(FakeTournament.params_schema, {"z": 1})
