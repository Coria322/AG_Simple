"""Registro de estrategias de operadores (RNF-04, RF-12).

Uso por decorador::

    @register_operator
    class TournamentSelection:
        name = "TOURNAMENT"
        kind = OperatorKind.SELECTION
        compatible_with = ALL_CHROMOSOME_TYPES
        params_schema = (ParamSpec("k", int, 3, minimum=2),)
        def select(self, fitness, n, rng, **params): ...

La clave es ``(kind, name)``: un cruce ``UNIFORM`` y una mutación ``UNIFORM``
conviven sin conflicto.
"""

from __future__ import annotations

import builtins
import threading
from typing import Any, TypeVar, overload

from plataforma_ag.core.operators.base import REQUIRED_METHOD, GeneticOperator
from plataforma_ag.domain.enums import ChromosomeType, OperatorKind
from plataforma_ag.domain.errors import DuplicateOperatorError, UnknownOperatorError

_T = TypeVar("_T")
_REQUIRED_ATTRS = ("name", "kind", "compatible_with", "params_schema")


class OperatorRegistry:
    """Catálogo de operadores consultable por familia y representación."""

    def __init__(self) -> None:
        self._operators: dict[tuple[OperatorKind, str], GeneticOperator] = {}
        self._lock = threading.Lock()

    # -- registro --------------------------------------------------------

    @overload
    def register(self, operator: type[_T]) -> type[_T]: ...
    @overload
    def register(self, operator: _T) -> _T: ...

    def register(self, operator: Any) -> Any:
        """Registra una clase (se instancia sin argumentos) o una instancia.

        Devuelve el mismo objeto recibido, para usarse como decorador.
        """
        instance = operator() if isinstance(operator, type) else operator
        self._check_contract(instance)
        key = (OperatorKind(instance.kind), str(instance.name))
        with self._lock:
            if key in self._operators:
                raise DuplicateOperatorError(
                    f"Ya existe un operador {key[0].value} con nombre '{key[1]}'"
                )
            self._operators[key] = instance
        return operator

    def unregister(self, kind: OperatorKind, name: str) -> None:
        with self._lock:
            if self._operators.pop((kind, name), None) is None:
                raise UnknownOperatorError(f"Operador {kind.value} '{name}' no registrado")

    # -- consulta --------------------------------------------------------

    def get(self, kind: OperatorKind, name: str) -> GeneticOperator:
        try:
            return self._operators[(kind, str(name))]
        except KeyError:
            raise UnknownOperatorError(f"Operador {kind.value} '{name}' no registrado") from None

    def list(
        self, kind: OperatorKind, chromosome_type: ChromosomeType | None = None
    ) -> builtins.list[GeneticOperator]:
        """Operadores de ``kind`` en orden de registro, filtrados por compatibilidad."""
        with self._lock:
            ops = [op for (k, _), op in self._operators.items() if k is kind]
        if chromosome_type is None:
            return ops
        return [op for op in ops if chromosome_type in op.compatible_with]

    def names(
        self, kind: OperatorKind, chromosome_type: ChromosomeType | None = None
    ) -> builtins.list[str]:
        return [op.name for op in self.list(kind, chromosome_type)]

    def __contains__(self, key: object) -> bool:
        return key in self._operators

    def __len__(self) -> int:
        return len(self._operators)

    # -- interno ---------------------------------------------------------

    @staticmethod
    def _check_contract(instance: object) -> None:
        missing = [attr for attr in _REQUIRED_ATTRS if not hasattr(instance, attr)]
        if missing:
            raise TypeError(
                f"{type(instance).__name__} no cumple el contrato de operador: "
                f"faltan {', '.join(missing)}"
            )
        kind = OperatorKind(getattr(instance, "kind"))  # noqa: B009 - validado arriba
        method = REQUIRED_METHOD[kind]
        if not callable(getattr(instance, method, None)):
            raise TypeError(
                f"{type(instance).__name__} es de tipo {kind.value} y debe implementar {method}()"
            )


operator_registry = OperatorRegistry()
"""Registro global de la aplicación (los operadores integrados se registran aquí)."""

register_operator = operator_registry.register
"""Decorador para registrar un operador en el registro global."""
