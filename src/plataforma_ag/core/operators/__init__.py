"""Strategy de operadores genéticos, registro y matriz de compatibilidad."""

from plataforma_ag.core.operators import (  # noqa: F401 - activa los decoradores de registro
    crossover,
    mutation,
    replacement,
    selection,
)
from plataforma_ag.core.operators.base import (
    CrossoverOperator,
    GeneticOperator,
    MutationContext,
    MutationOperator,
    ParamSpec,
    Population,
    ReplacementContext,
    ReplacementScheme,
    SelectionOperator,
)
from plataforma_ag.core.operators.compatibility import (
    COMPATIBILITY_MATRIX,
    CompatibilityRule,
    is_operator_allowed,
    validate_operator_config,
)
from plataforma_ag.core.operators.registry import (
    OperatorRegistry,
    operator_registry,
    register_operator,
)

__all__ = [
    "COMPATIBILITY_MATRIX",
    "CompatibilityRule",
    "CrossoverOperator",
    "GeneticOperator",
    "MutationContext",
    "MutationOperator",
    "OperatorRegistry",
    "ParamSpec",
    "Population",
    "ReplacementContext",
    "ReplacementScheme",
    "SelectionOperator",
    "is_operator_allowed",
    "operator_registry",
    "register_operator",
    "validate_operator_config",
]
