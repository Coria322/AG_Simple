"""Compatibilidad entre operadores y representaciones cromosómicas.

Re-exporta la definición canónica del dominio para compatibilidad hacia atrás en core.
"""

from plataforma_ag.domain.compatibility import (
    COMPATIBILITY_MATRIX,
    CompatibilityRule,
    get_default_operator,
    is_operator_allowed,
    validate_operator_config,
)

__all__ = [
    "COMPATIBILITY_MATRIX",
    "CompatibilityRule",
    "get_default_operator",
    "is_operator_allowed",
    "validate_operator_config",
]
