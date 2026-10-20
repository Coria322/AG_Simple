"""Servicio de exportación e importación de telemetría en CSV (RF-23, TASK-043).

Permite persistir y recargar los resultados de una corrida completa:
  - Columnas: `generation`, `individual_id`, `genotype`, `phenotype`, `fitness`, `is_elite`.
  - Reconstruye `RunHistoryStore` sin necesidad de reejecutar el algoritmo genético.
  - Valida la presencia de todas las columnas requeridas (lanza PersistenceError si faltan).
"""

from __future__ import annotations

import ast
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from plataforma_ag.domain.enums import ObjectiveSense
from plataforma_ag.domain.errors import PersistenceError
from plataforma_ag.domain.run import GenerationRecord, Individual
from plataforma_ag.services.history import RunHistoryStore

REQUIRED_CSV_COLUMNS = (
    "generation",
    "individual_id",
    "genotype",
    "phenotype",
    "fitness",
    "is_elite",
)


def _format_vector(vec: tuple[Any, ...] | list[Any]) -> str:
    """Formatea un vector para CSV entrecomillado o compacto."""
    # Si son bits simples (0 y 1)
    if all(isinstance(x, (int, np.integer)) and x in (0, 1) for x in vec):
        return "".join(str(int(x)) for x in vec)
    # Lista de flotantes o enteros
    return (
        "["
        + ", ".join(
            f"{float(x):.6g}" if isinstance(x, (float, np.floating)) else str(x) for x in vec
        )
        + "]"
    )


def _parse_vector_string(val_str: str) -> tuple[float | int, ...]:
    """Parsea una cadena de fenotipo o genotipo."""
    s = val_str.strip()
    # Caso 1: Cadena binaria simple como '01101'
    if s and all(c in "01" for c in s):
        return tuple(int(c) for c in s)

    # Caso 2: Notación de lista '[1.0, 2.0]' o tupla '(1.0, 2.0)'
    if s.startswith(("[", "(")) and s.endswith(("]", ")")):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return tuple(parsed)
        except (ValueError, SyntaxError):
            pass

    # Caso 3: Valores separados por comas o espacios
    parts = s.replace(",", " ").split()
    return tuple(float(p) if "." in p or "e" in p.lower() else int(p) for p in parts)


def export_run_to_csv(history: RunHistoryStore, filepath: Path | str) -> None:
    """Exporta todos los registros del historial a un archivo CSV."""
    path = Path(filepath)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(REQUIRED_CSV_COLUMNS)

            for gen in history.available_generations:
                rec = history.get_generation(gen)
                if rec is None:
                    continue
                for ind in rec.individuals:
                    geno_str = _format_vector(ind.genotype)
                    pheno_str = _format_vector(ind.phenotype)
                    fit_str = f"{ind.fitness:.8g}" if ind.fitness is not None else "nan"
                    elite_str = "1" if ind.is_elite else "0"

                    writer.writerow(
                        [
                            gen,
                            ind.individual_id,
                            geno_str,
                            pheno_str,
                            fit_str,
                            elite_str,
                        ]
                    )
    except OSError as err:
        raise PersistenceError(f"No se pudo exportar la telemetría a '{path}': {err}") from err


def import_run_from_csv(
    filepath: Path | str,
    sense: ObjectiveSense = ObjectiveSense.MINIMIZE,
) -> RunHistoryStore:
    """Importa telemetría desde CSV y reconstruye RunHistoryStore sin reejecutar el AG."""
    path = Path(filepath)
    if not path.is_file():
        raise PersistenceError(f"El archivo CSV no existe: '{path}'")

    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                headers = [h.strip() for h in next(reader)]
            except StopIteration as err:
                raise PersistenceError(f"El archivo CSV en '{path}' está vacío") from err

            # Validar columnas obligatorias (RF-23)
            for req in REQUIRED_CSV_COLUMNS:
                if req not in headers:
                    raise PersistenceError(
                        f"Falta la columna obligatoria '{req}' en CSV (cabeceras: {headers})"
                    )

            col_indices = {col: headers.index(col) for col in REQUIRED_CSV_COLUMNS}

            # Agrupar filas por generación
            generations_data: dict[int, list[Individual]] = {}

            for line_idx, row in enumerate(reader, start=2):
                if not row or not any(row):
                    continue
                try:
                    gen_num = int(row[col_indices["generation"]].strip())
                    ind_id = int(row[col_indices["individual_id"]].strip())
                    geno_str = row[col_indices["genotype"]].strip()
                    pheno_str = row[col_indices["phenotype"]].strip()
                    fit_raw = row[col_indices["fitness"]].strip()
                    elite_raw = row[col_indices["is_elite"]].strip()

                    fitness = float(fit_raw)
                    is_elite = elite_raw in ("1", "True", "true", "TRUE")
                    genotype = _parse_vector_string(geno_str)
                    phenotype = _parse_vector_string(pheno_str)

                    ind = Individual(
                        individual_id=ind_id,
                        genotype=genotype,
                        phenotype=tuple(float(x) for x in phenotype),
                        fitness=fitness,
                        is_elite=is_elite,
                    )

                    if gen_num not in generations_data:
                        generations_data[gen_num] = []
                    generations_data[gen_num].append(ind)

                except (ValueError, IndexError) as err:
                    raise PersistenceError(
                        f"Error de formato en la línea {line_idx} del CSV '{path}': {err}"
                    ) from err

    except OSError as err:
        raise PersistenceError(f"No se pudo leer el archivo CSV '{path}': {err}") from err

    # Reconstruir RunHistoryStore
    store = RunHistoryStore(sense=sense)

    for gen_num in sorted(generations_data.keys()):
        inds = generations_data[gen_num]
        finite_fits = [
            i.fitness for i in inds if i.fitness is not None and math.isfinite(i.fitness)
        ]

        if finite_fits:
            mean_f = float(sum(finite_fits) / len(finite_fits))
            best_f = float(
                min(finite_fits) if sense is ObjectiveSense.MINIMIZE else max(finite_fits)
            )
        else:
            mean_f = float("nan")
            best_f = float("inf") if sense is ObjectiveSense.MINIMIZE else float("-inf")

        record = GenerationRecord(
            generation=gen_num,
            best_fitness=best_f,
            mean_fitness=mean_f,
            elite_fitness_cumulative=best_f,
            individuals=tuple(inds),
        )
        store.add_generation(record)

    return store
