"""Almacén de historial de la corrida y cálculo de series de convergencia (RF-20, RF-21).

Gestiona los registros de generación (`GenerationRecord`), suministra vistas
paginadas para la tabla de individuos y calcula las tres curvas cartesianas:
  - Mejor fitness de la generación.
  - Promedio de la población (solo valores finitos).
  - Élite histórico acumulado (estrictamente monótono según el sentido de optimización).
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any

from plataforma_ag.domain.enums import ObjectiveSense
from plataforma_ag.domain.run import GenerationRecord, Individual


@dataclass(frozen=True, slots=True)
class ConvergenceSeries:
    """Series numéricas alineadas por generación para trazado cartesiano."""

    generations: tuple[int, ...]
    best_fitness: tuple[float, ...]
    mean_fitness: tuple[float, ...]
    cumulative_elite: tuple[float, ...]


class RunHistoryStore:
    """Almacén en memoria de registros generacionales con consultas paginadas."""

    def __init__(self, sense: ObjectiveSense = ObjectiveSense.MINIMIZE) -> None:
        self.sense = sense
        self._records: dict[int, GenerationRecord] = {}
        self._global_best: Individual | None = None
        self._global_best_gen: int = 0
        self._lock = threading.Lock()

    def clear(self) -> None:
        """Limpia todo el historial de la corrida."""
        with self._lock:
            self._records.clear()
            self._global_best = None
            self._global_best_gen = 0

    def add_generation(self, record: GenerationRecord) -> None:
        """Agrega o reemplaza un registro generacional de forma thread-safe."""
        with self._lock:
            self._records[record.generation] = record
            # Actualizar mejor solución global
            for ind in record.individuals:
                if ind.fitness is None or not math.isfinite(ind.fitness):
                    continue
                if self._global_best is None or self._global_best.fitness is None:
                    self._global_best = ind
                    self._global_best_gen = record.generation
                else:
                    if self.sense is ObjectiveSense.MINIMIZE:
                        if ind.fitness < self._global_best.fitness:
                            self._global_best = ind
                            self._global_best_gen = record.generation
                    else:
                        if ind.fitness > self._global_best.fitness:
                            self._global_best = ind
                            self._global_best_gen = record.generation

    def get_generation(self, generation: int) -> GenerationRecord | None:
        """Devuelve el registro de una generación específica o None si no existe."""
        with self._lock:
            return self._records.get(generation)

    @property
    def total_generations(self) -> int:
        """Cantidad de generaciones almacenadas."""
        with self._lock:
            return len(self._records)

    @property
    def available_generations(self) -> list[int]:
        """Lista ordenada de números de generación disponibles."""
        with self._lock:
            return sorted(self._records.keys())

    @property
    def global_best(self) -> Individual | None:
        """Mejor individuo histórico acumulado en toda la corrida."""
        with self._lock:
            return self._global_best

    @property
    def global_best_generation(self) -> int:
        """Generación en la cual se encontró el mejor individuo histórico."""
        with self._lock:
            return self._global_best_gen

    def get_individuals_page(
        self,
        generation: int,
        page: int = 1,
        page_size: int = 25,
        sort_by: str | None = None,
        sort_descending: bool = False,
    ) -> tuple[list[Individual], int, int]:
        """Consulta paginada de individuos de una generación.

        Args:
            generation: número de generación.
            page: página solicitada (1-based).
            page_size: cantidad de filas por página (mínimo 1).
            sort_by: campo de ordenamiento ('id', 'fitness', 'aptitude', 'is_elite').
            sort_descending: orden descendente si True.

        Returns:
            (individuos_de_la_pagina, total_individuos, total_paginas)
        """
        if page_size < 1:
            page_size = 25

        with self._lock:
            record = self._records.get(generation)
            if record is None or not record.individuals:
                return [], 0, 0

            individuals = list(record.individuals)

        # Ordenamiento
        if sort_by is not None:
            attr_name = "individual_id" if sort_by == "id" else sort_by

            def sort_key(ind: Individual) -> Any:
                val = getattr(ind, attr_name, None)
                if val is None:
                    return float("inf") if not sort_descending else float("-inf")
                return val

            individuals.sort(key=sort_key, reverse=sort_descending)

        total_count = len(individuals)
        total_pages = math.ceil(total_count / page_size)

        if page < 1:
            page = 1
        elif page > total_pages:
            return [], total_count, total_pages

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_items = individuals[start_idx:end_idx]

        return page_items, total_count, total_pages

    def get_convergence_series(self) -> ConvergenceSeries:
        """Calcula las series cartesianas de convergencia omitiendo o tratando no finitos.

        Garantiza que la curva de élite acumulado sea estrictamente monótona
        según el sentido de optimización (RF-21).
        """
        with self._lock:
            gens = sorted(self._records.keys())
            if not gens:
                return ConvergenceSeries((), (), (), ())

            best_list: list[float] = []
            mean_list: list[float] = []
            elite_list: list[float] = []

            current_elite = float("inf") if self.sense is ObjectiveSense.MINIMIZE else float("-inf")

            for g in gens:
                rec = self._records[g]
                best_f = rec.best_fitness
                mean_f = rec.mean_fitness

                best_list.append(best_f)
                mean_list.append(mean_f)

                # Actualizar élite acumulado monótonamente
                if math.isfinite(best_f):
                    if not math.isfinite(current_elite):
                        current_elite = best_f
                    elif self.sense is ObjectiveSense.MINIMIZE:
                        if best_f < current_elite:
                            current_elite = best_f
                    else:
                        if best_f > current_elite:
                            current_elite = best_f

                elite_list.append(current_elite)

            return ConvergenceSeries(
                generations=tuple(gens),
                best_fitness=tuple(best_list),
                mean_fitness=tuple(mean_list),
                cumulative_elite=tuple(elite_list),
            )
