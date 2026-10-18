"""Enumeraciones del dominio. Nombres y valores idénticos a ``DOMAIN_SCHEMA.json``."""

from __future__ import annotations

from enum import StrEnum


class _StrEnum(StrEnum):
    """Enum cuyo valor es la misma cadena del miembro (serializable a JSON tal cual)."""


class ChromosomeType(_StrEnum):
    """Representación cromosómica mostrada en la GUI (RF-11).

    AMB-003: en JSON, ``GRAY`` se serializa como ``BINARY`` + ``Encoding.GRAY``.
    """

    BINARY = "BINARY"
    GRAY = "GRAY"
    REAL = "REAL"
    PERMUTATION = "PERMUTATION"


class Encoding(_StrEnum):
    """Codificación de un cromosoma binario en el JSON (AMB-003)."""

    STANDARD = "STANDARD"
    GRAY = "GRAY"


class ObjectiveSense(_StrEnum):
    """Sentido de optimización (RF-04)."""

    MAXIMIZE = "MAXIMIZE"
    MINIMIZE = "MINIMIZE"


class FitnessTransform(_StrEnum):
    """Transformación de aptitud para minimizar maximizando (RF-04, AMB-001)."""

    NEGATE_PLUS_C = "NEGATE_PLUS_C"
    INVERSE_SHIFT = "INVERSE_SHIFT"


class AngleUnit(_StrEnum):
    """Unidad angular; solo radianes (RF-03)."""

    RADIANS = "RADIANS"


class BenchmarkFunction(_StrEnum):
    """Catálogo de funciones objetivo (RF-01) o expresión libre (RF-02)."""

    SPHERE = "SPHERE"
    RASTRIGIN = "RASTRIGIN"
    ROSENBROCK = "ROSENBROCK"
    ACKLEY = "ACKLEY"
    GRIEWANK = "GRIEWANK"
    CUSTOM_EXPRESSION = "CUSTOM_EXPRESSION"


class RedundantSpaceStrategy(_StrEnum):
    """Política de factibilidad del espacio redundante (RF-06, AMB-002)."""

    SATURATION = "SATURATION"
    REJECTION_RESAMPLE = "REJECTION_RESAMPLE"


class OperatorKind(_StrEnum):
    """Familia de operador genético (RNF-04). No figura en el ERS: organiza el registro."""

    SELECTION = "SELECTION"
    CROSSOVER = "CROSSOVER"
    MUTATION = "MUTATION"
    REPLACEMENT = "REPLACEMENT"


class SelectionMethod(_StrEnum):
    """Métodos de selección (RF-13)."""

    ROULETTE = "ROULETTE"
    STOCHASTIC_UNIVERSAL_SAMPLING = "STOCHASTIC_UNIVERSAL_SAMPLING"
    LINEAR_RANKING = "LINEAR_RANKING"
    NONLINEAR_RANKING = "NONLINEAR_RANKING"
    TOURNAMENT = "TOURNAMENT"


class CrossoverMethod(_StrEnum):
    """Métodos de cruce (RF-14)."""

    SINGLE_POINT = "SINGLE_POINT"
    UNIFORM = "UNIFORM"
    ARITHMETIC = "ARITHMETIC"
    FLAT = "FLAT"
    BLX_ALPHA = "BLX_ALPHA"
    PMX = "PMX"
    OX = "OX"
    CX = "CX"


class MutationMethod(_StrEnum):
    """Métodos de mutación (RF-15)."""

    SIMPLE_FLIP = "SIMPLE_FLIP"
    UNIFORM = "UNIFORM"
    NON_UNIFORM = "NON_UNIFORM"
    SWAP = "SWAP"
    INVERSION = "INVERSION"


class ReplacementMethod(_StrEnum):
    """Esquemas de reemplazo (RF-16). AMB-005: elitismo combinable vía ``elitism_count``."""

    GENERATIONAL = "GENERATIONAL"
    ELITISM = "ELITISM"
    STEADY_STATE = "STEADY_STATE"
    CROWDING = "CROWDING"


class RunState(_StrEnum):
    """Estados de la máquina de control de flujo (RF-17)."""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


class StopReason(_StrEnum):
    """Razón de finalización de una corrida (RF-09, RF-17)."""

    MAX_GENERATIONS = "MAX_GENERATIONS"
    STAGNATION = "STAGNATION"
    USER_ABORT = "USER_ABORT"
    ERROR = "ERROR"


class LogLevel(_StrEnum):
    """Niveles de la bitácora (RF-19)."""

    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
