"""Reglas de validación integral del experimento y formularios (RF-10, TASK-036).

Valida reglas numéricas, de dominio, de operadores y de criterios de parada.
Reutilizable tanto por la GUI como por la importación de archivos JSON.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from plataforma_ag.domain.compatibility import is_operator_allowed
from plataforma_ag.domain.config import ExperimentConfig
from plataforma_ag.domain.enums import (
    BenchmarkFunction,
    ChromosomeType,
    OperatorKind,
)

BENCHMARK_MIN_DIMENSIONS: dict[BenchmarkFunction, int] = {
    BenchmarkFunction.ROSENBROCK: 2,
}


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Resultado estructurado de validación."""

    is_valid: bool
    errors: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return not self.is_valid or len(self.errors) > 0


_COMMA_DECIMAL_RE = re.compile(r"^\s*[-+]?\d+,\d+\s*$")


def validate_number_format(text: str, integer_only: bool = False) -> str | None:
    """Valida el formato textual de un número.

    Devuelve None si es válido o un mensaje de error legible.
    """
    stripped = text.strip()
    if not stripped:
        return "El campo no puede estar vacío"

    if _COMMA_DECIMAL_RE.match(stripped):
        return "Formato numérico inválido: usa punto decimal (.) en lugar de coma (,)"

    try:
        if integer_only:
            val_int = int(stripped)
            if str(val_int) != stripped and not stripped.startswith("+"):
                return "Debe ser un número entero"
        else:
            float(stripped)
    except ValueError:
        return "Debe ser un número entero" if integer_only else "Debe ser un valor numérico válido"

    return None


def validate_experiment_config(config: ExperimentConfig) -> ValidationResult:
    """Valida exhaustivamente una configuración completa de experimento."""
    errors: dict[str, str] = {}
    warnings: list[str] = []

    # 1. Validación del problema
    prob = config.problem
    if prob.dimensions < 1:
        errors["dimensions"] = "Las dimensiones deben ser un entero >= 1"

    if prob.benchmark is not BenchmarkFunction.CUSTOM_EXPRESSION:
        min_dim = BENCHMARK_MIN_DIMENSIONS.get(prob.benchmark, 1)
        if prob.dimensions < min_dim:
            errors["dimensions"] = (
                f"El benchmark '{prob.benchmark.value}' requiere al menos {min_dim} dimensiones "
                f"(se configuró {prob.dimensions})"
            )
    else:
        if not prob.expression or not prob.expression.strip():
            errors["expression"] = "La expresión matemática personalizada no puede estar vacía"

    # 2. Validación de dominios por variable
    if len(prob.bounds) != prob.dimensions:
        errors["bounds"] = (
            f"Se esperaban límites para {prob.dimensions} variables, pero hay {len(prob.bounds)}"
        )

    var_names_seen: set[str] = set()
    for idx, b in enumerate(prob.bounds):
        key_prefix = f"bounds.{b.variable or f'idx_{idx}'}"
        if b.variable in var_names_seen:
            errors[key_prefix] = f"Variable '{b.variable}' duplicada en los límites"
        var_names_seen.add(b.variable)

        if b.min >= b.max:
            errors[f"{key_prefix}.range"] = (
                f"Dominio invertido en {b.variable}: min ({b.min}) debe ser menor que max ({b.max})"
            )

        if config.genetic.chromosome_type in (
            ChromosomeType.BINARY,
            ChromosomeType.GRAY,
        ) and (b.bits is None or b.bits < 1 or b.bits > 64):
            errors[f"{key_prefix}.bits"] = (
                f"Los bits para {b.variable} deben estar en [1, 64] (recibido {b.bits})"
            )

    # 3. Validación de parámetros genéticos
    gen = config.genetic
    if gen.population_size < 2:
        errors["population_size"] = "El tamaño de la población (Np) debe ser al menos 2"

    if not (0.0 <= gen.p_crossover <= 1.0):
        errors["p_crossover"] = (
            f"Probabilidad de cruce (Pc) debe estar en [0, 1] (recibido {gen.p_crossover})"
        )

    if not (0.0 <= gen.p_mutation <= 1.0):
        errors["p_mutation"] = (
            f"Probabilidad de mutación (Pm) debe estar en [0, 1] (recibido {gen.p_mutation})"
        )

    # 4. Compatibilidad de operadores con el cromosoma
    chrom = gen.chromosome_type
    ops = config.operators

    if not is_operator_allowed(chrom, OperatorKind.SELECTION, ops.selection_method.value):
        errors["selection_method"] = (
            f"Selección '{ops.selection_method.value}' incompatible con '{chrom.value}'"
        )
    if not is_operator_allowed(chrom, OperatorKind.CROSSOVER, ops.crossover_method.value):
        errors["crossover_method"] = (
            f"Cruce '{ops.crossover_method.value}' incompatible con '{chrom.value}'"
        )
    if not is_operator_allowed(chrom, OperatorKind.MUTATION, ops.mutation_method.value):
        errors["mutation_method"] = (
            f"Mutación '{ops.mutation_method.value}' incompatible con '{chrom.value}'"
        )
    if not is_operator_allowed(chrom, OperatorKind.REPLACEMENT, ops.replacement_method.value):
        errors["replacement_method"] = (
            f"Reemplazo '{ops.replacement_method.value}' incompatible con '{chrom.value}'"
        )

    if ops.elitism_count is not None and ops.elitism_count < 0:
        errors["elitism_count"] = "El conteo de individuos élite (E) debe ser >= 0"

    # 5. Validación de criterios de parada
    stop = config.stopping
    if stop.max_generations < 1:
        errors["max_generations"] = "El número máximo de generaciones (Gmax) debe ser >= 1"

    if stop.stagnation_enabled:
        if stop.stagnation_generations < 1:
            errors["stagnation_generations"] = (
                "Las generaciones de estancamiento deben ser un entero >= 1"
            )
        if stop.stagnation_epsilon < 0.0:
            errors["stagnation_epsilon"] = "El umbral epsilon de estancamiento debe ser >= 0.0"

    is_valid = len(errors) == 0
    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


def validate_single_field(field_name: str, value: Any) -> str | None:
    """Valida un campo individual y devuelve el mensaje de error o None."""
    if field_name == "p_crossover":
        if isinstance(value, (int, float)):
            if not (0.0 <= float(value) <= 1.0):
                return "La probabilidad de cruce (Pc) debe estar entre 0.0 y 1.0"
        elif isinstance(value, str):
            err = validate_number_format(value)
            if err:
                return err
            val = float(value)
            if not (0.0 <= val <= 1.0):
                return "La probabilidad de cruce (Pc) debe estar entre 0.0 y 1.0"

    elif field_name == "p_mutation":
        if isinstance(value, (int, float)):
            if not (0.0 <= float(value) <= 1.0):
                return "La probabilidad de mutación (Pm) debe estar entre 0.0 y 1.0"
        elif isinstance(value, str):
            err = validate_number_format(value)
            if err:
                return err
            val = float(value)
            if not (0.0 <= val <= 1.0):
                return "La probabilidad de mutación (Pm) debe estar entre 0.0 y 1.0"

    elif field_name == "population_size":
        if isinstance(value, (int, float)):
            if int(value) < 2:
                return "El tamaño de la población (Np) debe ser al menos 2"
        elif isinstance(value, str):
            err = validate_number_format(value, integer_only=True)
            if err:
                return err
            if int(value) < 2:
                return "El tamaño de la población (Np) debe ser al menos 2"

    elif field_name == "max_generations":
        if isinstance(value, (int, float)):
            if int(value) < 1:
                return "El número máximo de generaciones (Gmax) debe ser >= 1"
        elif isinstance(value, str):
            err = validate_number_format(value, integer_only=True)
            if err:
                return err
            if int(value) < 1:
                return "El número máximo de generaciones (Gmax) debe ser >= 1"

    elif field_name == "dimensions":
        if isinstance(value, (int, float)):
            if int(value) < 1:
                return "Las dimensiones deben ser un entero >= 1"
        elif isinstance(value, str):
            err = validate_number_format(value, integer_only=True)
            if err:
                return err
            if int(value) < 1:
                return "Las dimensiones deben ser un entero >= 1"

    return None
