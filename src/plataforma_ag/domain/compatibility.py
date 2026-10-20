"""Matriz de compatibilidad entre representaciones y operadores (RF-12).

Define declarativamente como datos (no lógica dispersa) qué operadores de selección,
cruce y mutación están habilitados para cada tipo de cromosoma.
"""

from __future__ import annotations

from dataclasses import dataclass

from plataforma_ag.domain.config import OperatorConfig
from plataforma_ag.domain.enums import (
    ChromosomeType,
    CrossoverMethod,
    MutationMethod,
    OperatorKind,
    ReplacementMethod,
    SelectionMethod,
)
from plataforma_ag.domain.errors import IncompatibleOperatorError


@dataclass(frozen=True, slots=True)
class CompatibilityRule:
    """Regla declarativa de operadores permitidos para una representación."""

    chromosome_type: ChromosomeType
    allowed_crossover: frozenset[CrossoverMethod]
    allowed_mutation: frozenset[MutationMethod]
    allowed_selection: frozenset[SelectionMethod]
    allowed_replacement: frozenset[ReplacementMethod] = frozenset(ReplacementMethod)


# Matriz declarativa conforme a la tabla del ERS RF-12
COMPATIBILITY_MATRIX: dict[ChromosomeType, CompatibilityRule] = {
    ChromosomeType.BINARY: CompatibilityRule(
        chromosome_type=ChromosomeType.BINARY,
        allowed_crossover=frozenset(
            {
                CrossoverMethod.SINGLE_POINT,
                CrossoverMethod.UNIFORM,
            }
        ),
        allowed_mutation=frozenset(
            {
                MutationMethod.SIMPLE_FLIP,
                MutationMethod.UNIFORM,
                MutationMethod.NON_UNIFORM,
            }
        ),
        allowed_selection=frozenset(
            {
                SelectionMethod.ROULETTE,
                SelectionMethod.STOCHASTIC_UNIVERSAL_SAMPLING,
                SelectionMethod.LINEAR_RANKING,
                SelectionMethod.NONLINEAR_RANKING,
                SelectionMethod.TOURNAMENT,
            }
        ),
    ),
    ChromosomeType.GRAY: CompatibilityRule(
        chromosome_type=ChromosomeType.GRAY,
        allowed_crossover=frozenset(
            {
                CrossoverMethod.SINGLE_POINT,
                CrossoverMethod.UNIFORM,
            }
        ),
        allowed_mutation=frozenset(
            {
                MutationMethod.SIMPLE_FLIP,
                MutationMethod.UNIFORM,
                MutationMethod.NON_UNIFORM,
            }
        ),
        allowed_selection=frozenset(
            {
                SelectionMethod.ROULETTE,
                SelectionMethod.STOCHASTIC_UNIVERSAL_SAMPLING,
                SelectionMethod.LINEAR_RANKING,
                SelectionMethod.NONLINEAR_RANKING,
                SelectionMethod.TOURNAMENT,
            }
        ),
    ),
    ChromosomeType.REAL: CompatibilityRule(
        chromosome_type=ChromosomeType.REAL,
        allowed_crossover=frozenset(
            {
                CrossoverMethod.UNIFORM,
                CrossoverMethod.ARITHMETIC,
                CrossoverMethod.FLAT,
                CrossoverMethod.BLX_ALPHA,
            }
        ),
        allowed_mutation=frozenset(
            {
                MutationMethod.UNIFORM,
                MutationMethod.NON_UNIFORM,
            }
        ),
        allowed_selection=frozenset(
            {
                SelectionMethod.ROULETTE,
                SelectionMethod.STOCHASTIC_UNIVERSAL_SAMPLING,
                SelectionMethod.LINEAR_RANKING,
                SelectionMethod.NONLINEAR_RANKING,
                SelectionMethod.TOURNAMENT,
            }
        ),
    ),
    ChromosomeType.PERMUTATION: CompatibilityRule(
        chromosome_type=ChromosomeType.PERMUTATION,
        allowed_crossover=frozenset(
            {
                CrossoverMethod.PMX,
                CrossoverMethod.OX,
                CrossoverMethod.CX,
            }
        ),
        allowed_mutation=frozenset(
            {
                MutationMethod.SWAP,
                MutationMethod.INVERSION,
            }
        ),
        # Ranking no lineal deshabilitado en permutaciones según tabla RF-12
        allowed_selection=frozenset(
            {
                SelectionMethod.ROULETTE,
                SelectionMethod.STOCHASTIC_UNIVERSAL_SAMPLING,
                SelectionMethod.LINEAR_RANKING,
                SelectionMethod.TOURNAMENT,
            }
        ),
    ),
}


def is_operator_allowed(
    chromosome_type: ChromosomeType, kind: OperatorKind, method_name: str
) -> bool:
    """Verifica si un operador específico está permitido para el tipo de cromosoma."""
    rule = COMPATIBILITY_MATRIX.get(chromosome_type)
    if rule is None:
        return False

    if kind is OperatorKind.CROSSOVER:
        return method_name in {m.value for m in rule.allowed_crossover}
    if kind is OperatorKind.MUTATION:
        return method_name in {m.value for m in rule.allowed_mutation}
    if kind is OperatorKind.SELECTION:
        return method_name in {m.value for m in rule.allowed_selection}
    if kind is OperatorKind.REPLACEMENT:
        return method_name in {m.value for m in rule.allowed_replacement}

    return False


def validate_operator_config(chromosome_type: ChromosomeType, operators: OperatorConfig) -> None:
    """Valida la configuración completa de operadores contra la matriz de compatibilidad."""
    rule = COMPATIBILITY_MATRIX.get(chromosome_type)
    if rule is None:
        raise IncompatibleOperatorError(f"Tipo de cromosoma desconocido: {chromosome_type}")

    if operators.selection_method not in rule.allowed_selection:
        raise IncompatibleOperatorError(
            f"El método de selección '{operators.selection_method.value}' "
            f"no es compatible con la representación {chromosome_type.value}."
        )

    if operators.crossover_method not in rule.allowed_crossover:
        raise IncompatibleOperatorError(
            f"El método de cruce '{operators.crossover_method.value}' "
            f"no es compatible con la representación {chromosome_type.value}."
        )

    if operators.mutation_method not in rule.allowed_mutation:
        raise IncompatibleOperatorError(
            f"El método de mutación '{operators.mutation_method.value}' "
            f"no es compatible con la representación {chromosome_type.value}."
        )

    if operators.replacement_method not in rule.allowed_replacement:
        raise IncompatibleOperatorError(
            f"El esquema de reemplazo '{operators.replacement_method.value}' "
            f"no es compatible con la representación {chromosome_type.value}."
        )


def get_default_operator(chromosome_type: ChromosomeType, kind: OperatorKind) -> str:
    """Devuelve un operador válido por defecto para una representación y tipo dado."""
    rule = COMPATIBILITY_MATRIX[chromosome_type]
    if kind is OperatorKind.CROSSOVER:
        if CrossoverMethod.SINGLE_POINT in rule.allowed_crossover:
            return CrossoverMethod.SINGLE_POINT.value
        if CrossoverMethod.PMX in rule.allowed_crossover:
            return CrossoverMethod.PMX.value
        return CrossoverMethod.UNIFORM.value
    if kind is OperatorKind.MUTATION:
        if MutationMethod.SIMPLE_FLIP in rule.allowed_mutation:
            return MutationMethod.SIMPLE_FLIP.value
        if MutationMethod.SWAP in rule.allowed_mutation:
            return MutationMethod.SWAP.value
        return MutationMethod.UNIFORM.value
    if kind is OperatorKind.SELECTION:
        return SelectionMethod.TOURNAMENT.value
    return ReplacementMethod.GENERATIONAL.value
