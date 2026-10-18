"""Errores del dominio (ARCHITECTURE.md § Manejo de errores)."""

from __future__ import annotations


class PlataformaAGError(Exception):
    """Raíz de todas las excepciones propias de la plataforma."""


class ExpressionError(PlataformaAGError):
    """Expresión matemática inválida, insegura o en grados (RF-02, RF-03).

    ``position`` es la columna aproximada (base 1) en el texto original, si se conoce.
    """

    def __init__(self, message: str, position: int | None = None) -> None:
        self.message = message
        self.position = position
        suffix = f" (posición {position})" if position is not None else ""
        super().__init__(f"{message}{suffix}")


class ValidationError(PlataformaAGError):
    """Parámetro fuera de rango o dominio inválido (RF-10)."""

    def __init__(self, message: str, field: str | None = None) -> None:
        self.message = message
        self.field = field
        prefix = f"{field}: " if field else ""
        super().__init__(f"{prefix}{message}")


class IncompatibleOperatorError(PlataformaAGError):
    """Operador no permitido para la representación activa (RF-12)."""


class DuplicateOperatorError(PlataformaAGError):
    """Se intentó registrar dos operadores con el mismo (tipo, nombre) (RNF-04)."""


class UnknownOperatorError(PlataformaAGError, LookupError):
    """Se pidió un operador no registrado."""


class InvalidStateTransition(PlataformaAGError):  # noqa: N818 - nombre fijado por ARCHITECTURE.md
    """Transición inválida de la máquina de estados de la corrida (RF-17)."""

    def __init__(self, current: object, requested: str) -> None:
        self.current = current
        self.requested = requested
        super().__init__(f"No se puede '{requested}' desde el estado {current}")


class PersistenceError(PlataformaAGError):
    """Archivo JSON/CSV/PDF ilegible o no escribible (RF-22..RF-24)."""


class MathFault(PlataformaAGError):  # noqa: N818 - nombre fijado por ARCHITECTURE.md
    """Fallo matemático interno (÷0, √ negativa, overflow) que se penaliza (RNF-03)."""
