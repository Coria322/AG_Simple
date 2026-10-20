"""Servicio de persistencia de configuración en formato JSON (RF-22, TASK-042).

Exporta e importa la configuración completa del experimento según ERS §4.1:
  - `project_metadata`: nombre y marca de tiempo en ISO-8601 UTC.
  - `problem_configuration`: función objetivo, dimensiones, límites, unidad angular, sentido.
  - `genetic_parameters`: población, cromosoma, codificación, Pc, Pm y semilla (AMB-008).
  - `operators`: métodos y parámetros de selección, cruce, mutación, reemplazo y élite.
  - `stopping_criteria`: máximo de generaciones (Gmax) y estancamiento opcional.

Aplica validación de compatibilidad (RF-12) y de dominio (RF-10) antes de aplicar cambios.
"""

from __future__ import annotations

import json
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from plataforma_ag.domain.compatibility import validate_operator_config
from plataforma_ag.domain.config import (
    ExperimentConfig,
    GeneticParameters,
    OperatorConfig,
    ProblemDefinition,
    StoppingCriteria,
    VariableBound,
)
from plataforma_ag.domain.enums import (
    AngleUnit,
    BenchmarkFunction,
    ChromosomeType,
    CrossoverMethod,
    FitnessTransform,
    MutationMethod,
    ObjectiveSense,
    RedundantSpaceStrategy,
    ReplacementMethod,
    SelectionMethod,
)
from plataforma_ag.domain.errors import (
    IncompatibleOperatorError,
    PersistenceError,
    ValidationError,
)
from plataforma_ag.domain.validation import validate_experiment_config


def export_config_to_json(
    config: ExperimentConfig,
    filepath: Path | str,
    project_name: str = "Experimento AG",
) -> None:
    """Exporta la configuración completa a un archivo JSON en UTF-8 con indentación."""
    path = Path(filepath)

    # Serializar límites
    bounds_data = [
        {
            "variable": b.variable,
            "min": b.min,
            "max": b.max,
            "bits": b.bits,
        }
        for b in config.problem.bounds
    ]

    # AMB-003: Cromosoma Gray se serializa como BINARY + encoding=GRAY
    chrom_type_str = config.genetic.chromosome_type.value
    encoding_str = "STANDARD"
    if config.genetic.chromosome_type is ChromosomeType.GRAY:
        chrom_type_str = ChromosomeType.BINARY.value
        encoding_str = "GRAY"

    data: dict[str, Any] = {
        "schema_version": "1.0",
        "project_metadata": {
            "name": project_name,
            "timestamp": datetime.now(UTC).isoformat(),
        },
        "problem_configuration": {
            "benchmark": config.problem.benchmark.value,
            "dimensions": config.problem.dimensions,
            "expression": config.problem.expression,
            "angle_unit": config.problem.angle_unit.value,
            "objective": config.problem.objective.value,
            "redundant_space_strategy": config.problem.redundant_space_strategy.value,
            "fitness_transform": config.problem.fitness_transform.value,
            "bounds": bounds_data,
        },
        "genetic_parameters": {
            "population_size": config.genetic.population_size,
            "chromosome_type": chrom_type_str,
            "encoding": encoding_str,
            "p_crossover": config.genetic.p_crossover,
            "p_mutation": config.genetic.p_mutation,
            "seed": config.genetic.seed,
        },
        "operators": {
            "selection_method": config.operators.selection_method.value,
            "selection_params": dict(config.operators.selection_params),
            "crossover_method": config.operators.crossover_method.value,
            "crossover_params": dict(config.operators.crossover_params),
            "mutation_method": config.operators.mutation_method.value,
            "mutation_params": dict(config.operators.mutation_params),
            "replacement_method": config.operators.replacement_method.value,
            "elitism_count": config.operators.elitism_count,
            "n_replace": config.operators.n_replace,
            "crowding_factor": config.operators.crowding_factor,
        },
        "stopping_criteria": {
            "max_generations": config.stopping.max_generations,
            "stagnation_enabled": config.stopping.stagnation_enabled,
            "stagnation_generations": config.stopping.stagnation_generations,
            "stagnation_epsilon": config.stopping.stagnation_epsilon,
        },
    }

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as err:
        raise PersistenceError(f"No se pudo guardar la configuración en '{path}': {err}") from err


def import_config_from_json(filepath: Path | str) -> ExperimentConfig:
    """Lee y valida un archivo de configuración JSON restaurando ExperimentConfig.

    Lanza:
      - PersistenceError: si el archivo no existe, no es JSON válido o faltan secciones.
      - IncompatibleOperatorError: si la matriz de operadores es incompatible con el cromosoma.
      - ValidationError: si algún parámetro numérico o dominio es inválido.
    """
    path = Path(filepath)
    if not path.is_file():
        raise PersistenceError(f"El archivo de configuración no existe: '{path}'")

    try:
        with open(path, encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as err:
        raise PersistenceError(f"Archivo JSON corrupto o inválido en '{path}': {err}") from err
    except OSError as err:
        raise PersistenceError(f"No se pudo leer el archivo '{path}': {err}") from err

    if not isinstance(raw_data, dict):
        raise PersistenceError("El JSON raíz debe ser un objeto")

    # Identificar claves desconocidas
    known_root_keys = {
        "schema_version",
        "project_metadata",
        "problem_configuration",
        "genetic_parameters",
        "operators",
        "stopping_criteria",
    }
    unknown_keys = set(raw_data.keys()) - known_root_keys
    if unknown_keys:
        warnings.warn(
            f"Claves desconocidas ignoradas en JSON: {sorted(unknown_keys)}",
            UserWarning,
            stacklevel=2,
        )

    prob_data = raw_data.get("problem_configuration")
    gen_data = raw_data.get("genetic_parameters")
    ops_data = raw_data.get("operators")
    stop_data = raw_data.get("stopping_criteria")

    if not (
        isinstance(prob_data, dict)
        and isinstance(gen_data, dict)
        and isinstance(ops_data, dict)
        and isinstance(stop_data, dict)
    ):
        raise PersistenceError(
            "El archivo JSON carece de alguna de las secciones obligatorias: "
            "'problem_configuration', 'genetic_parameters', 'operators', 'stopping_criteria'"
        )

    try:
        # 1. Parámetros genéticos
        chrom_raw = str(gen_data.get("chromosome_type", "BINARY"))
        enc_raw = str(gen_data.get("encoding", "STANDARD"))
        # AMB-003: Reconstruir ChromosomeType.GRAY si es BINARY + GRAY
        if chrom_raw == "BINARY" and enc_raw == "GRAY":
            chrom_type = ChromosomeType.GRAY
        else:
            chrom_type = ChromosomeType(chrom_raw)

        seed_val = gen_data.get("seed")
        genetic = GeneticParameters(
            population_size=int(gen_data.get("population_size", 100)),
            chromosome_type=chrom_type,
            p_crossover=float(gen_data.get("p_crossover", 0.85)),
            p_mutation=float(gen_data.get("p_mutation", 0.05)),
            seed=int(seed_val) if seed_val is not None else None,
        )

        # 2. Operadores
        # Soporte para formato plano o anidado
        sel_method_raw = ops_data.get("selection_method") or ops_data.get("selection", {}).get(
            "method"
        )
        sel_params = ops_data.get("selection_params") or ops_data.get("selection", {}).get(
            "parameters", {}
        )

        cross_method_raw = ops_data.get("crossover_method") or ops_data.get("crossover", {}).get(
            "method"
        )
        cross_params = ops_data.get("crossover_params") or ops_data.get("crossover", {}).get(
            "parameters", {}
        )

        mut_method_raw = ops_data.get("mutation_method") or ops_data.get("mutation", {}).get(
            "method"
        )
        mut_params = ops_data.get("mutation_params") or ops_data.get("mutation", {}).get(
            "parameters", {}
        )

        rep_method_raw = ops_data.get("replacement_method") or ops_data.get("replacement", {}).get(
            "method", "GENERATIONAL"
        )

        operators = OperatorConfig(
            selection_method=SelectionMethod(str(sel_method_raw)),
            selection_params=dict(sel_params or {}),
            crossover_method=CrossoverMethod(str(cross_method_raw)),
            crossover_params=dict(cross_params or {}),
            mutation_method=MutationMethod(str(mut_method_raw)),
            mutation_params=dict(mut_params or {}),
            replacement_method=ReplacementMethod(str(rep_method_raw)),
            elitism_count=ops_data.get("elitism_count"),
            n_replace=ops_data.get("n_replace"),
            crowding_factor=ops_data.get("crowding_factor"),
        )

        # Validar compatibilidad de operadores antes de continuar (RF-12, RF-22)
        try:
            validate_operator_config(chrom_type, operators)
        except IncompatibleOperatorError as err:
            raise IncompatibleOperatorError(
                f"Configuración de operadores incompatible con cromosoma {chrom_type.value}: {err}"
            ) from err

        # 3. Límites de variables
        bounds_raw = prob_data.get("bounds", [])
        bounds_list: list[VariableBound] = []
        for b in bounds_raw:
            bits_val = b.get("bits")
            bounds_list.append(
                VariableBound(
                    variable=str(b["variable"]),
                    min=float(b["min"]),
                    max=float(b["max"]),
                    bits=int(bits_val) if bits_val is not None else 16,
                )
            )

        # 4. Definición del problema
        benchmark_raw = prob_data.get("benchmark", "CUSTOM_EXPRESSION")
        problem = ProblemDefinition(
            expression=str(prob_data.get("expression") or ""),
            dimensions=int(prob_data.get("dimensions", len(bounds_list) or 1)),
            bounds=tuple(bounds_list),
            benchmark=BenchmarkFunction(str(benchmark_raw)),
            angle_unit=AngleUnit(str(prob_data.get("angle_unit", "RADIANS"))),
            objective=ObjectiveSense(str(prob_data.get("objective", "MINIMIZE"))),
            redundant_space_strategy=RedundantSpaceStrategy(
                str(prob_data.get("redundant_space_strategy", "SATURATION"))
            ),
            fitness_transform=FitnessTransform(
                str(prob_data.get("fitness_transform", "NEGATE_PLUS_C"))
            ),
        )

        # 5. Criterios de parada
        stagnation_data = stop_data.get("stagnation")
        if isinstance(stagnation_data, dict):
            stag_enabled = bool(stagnation_data.get("enabled", False))
            stag_gens = int(stagnation_data.get("generations", 25))
            stag_eps = float(stagnation_data.get("epsilon", 1e-6))
        else:
            stag_enabled = bool(stop_data.get("stagnation_enabled", False))
            stag_gens = int(stop_data.get("stagnation_generations", 25))
            stag_eps = float(stop_data.get("stagnation_epsilon", 1e-6))

        stopping = StoppingCriteria(
            max_generations=int(stop_data.get("max_generations", 200)),
            stagnation_enabled=stag_enabled,
            stagnation_generations=stag_gens,
            stagnation_epsilon=stag_eps,
        )

        config = ExperimentConfig(
            problem=problem,
            genetic=genetic,
            operators=operators,
            stopping=stopping,
        )

        # Validación integral de la configuración resultante (RF-10)
        v_res = validate_experiment_config(config)
        if v_res.has_errors:
            err_details = "; ".join(f"{k}: {v}" for k, v in v_res.errors.items())
            raise ValidationError(f"Configuración inválida en JSON: {err_details}")

        return config

    except (ValueError, KeyError, TypeError) as err:
        raise PersistenceError(f"Error interpretando la configuración JSON: {err}") from err
