"""Controlador de parámetros genéticos y operadores (RF-07..RF-16, TASK-024..026).

Gestiona la representación cromosómica, población, probabilidades Pc y Pm,
criterios de parada, esquemas de operadores y la matriz de compatibilidad.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

from plataforma_ag.domain.compatibility import (
    COMPATIBILITY_MATRIX,
)
from plataforma_ag.domain.config import (
    DEFAULT_MAX_GENERATIONS,
    DEFAULT_STAGNATION_EPSILON,
    DEFAULT_STAGNATION_GENERATIONS,
    GeneticParameters,
    OperatorConfig,
    StoppingCriteria,
)
from plataforma_ag.domain.enums import (
    ChromosomeType,
    CrossoverMethod,
    MutationMethod,
    ReplacementMethod,
    SelectionMethod,
)


class ParamsController:
    """Controlador que coordina los parámetros de simulación y operadores."""

    def __init__(
        self,
        initial_genetic: GeneticParameters | None = None,
        initial_operators: OperatorConfig | None = None,
        initial_stopping: StoppingCriteria | None = None,
    ) -> None:
        self._genetic = initial_genetic or GeneticParameters()
        self._operators = initial_operators or OperatorConfig()
        self._stopping = initial_stopping or StoppingCriteria()

        self._listeners: list[Callable[[], None]] = []
        self._operator_changed_notifications: list[Callable[[str, str], None]] = []

    @property
    def genetic(self) -> GeneticParameters:
        return self._genetic

    @property
    def operators(self) -> OperatorConfig:
        return self._operators

    @property
    def stopping(self) -> StoppingCriteria:
        return self._stopping

    def add_listener(self, callback: Callable[[], None]) -> None:
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[], None]) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    def add_operator_notification_listener(self, callback: Callable[[str, str], None]) -> None:
        """Listener para alertar al usuario cuando un operador cambia automáticamente (toast)."""
        if callback not in self._operator_changed_notifications:
            self._operator_changed_notifications.append(callback)

    def _notify(self) -> None:
        for cb in list(self._listeners):
            with contextlib.suppress(Exception):
                cb()

    def _notify_operator_change(self, kind: str, new_operator: str) -> None:
        for cb in list(self._operator_changed_notifications):
            with contextlib.suppress(Exception):
                cb(kind, new_operator)

    def set_chromosome_type(self, new_type: ChromosomeType) -> None:
        """Cambia el tipo de cromosoma y asegura que los operadores sigan siendo compatibles."""
        if self._genetic.chromosome_type == new_type:
            return

        self._genetic = GeneticParameters(
            population_size=self._genetic.population_size,
            chromosome_type=new_type,
            p_crossover=self._genetic.p_crossover,
            p_mutation=self._genetic.p_mutation,
            seed=self._genetic.seed,
        )

        # Verificar y ajustar operadores si son incompatibles
        rule = COMPATIBILITY_MATRIX[new_type]
        sel = self._operators.selection_method
        cx = self._operators.crossover_method
        mut = self._operators.mutation_method
        rep = self._operators.replacement_method

        if sel not in rule.allowed_selection:
            sel = next(iter(rule.allowed_selection))
            self._notify_operator_change("Selección", sel.value)

        if cx not in rule.allowed_crossover:
            cx = next(iter(rule.allowed_crossover))
            self._notify_operator_change("Cruce", cx.value)

        if mut not in rule.allowed_mutation:
            mut = next(iter(rule.allowed_mutation))
            self._notify_operator_change("Mutación", mut.value)

        if rep not in rule.allowed_replacement:
            rep = next(iter(rule.allowed_replacement))
            self._notify_operator_change("Reemplazo", rep.value)

        self._operators = OperatorConfig(
            selection_method=sel,
            selection_params=dict(self._operators.selection_params),
            crossover_method=cx,
            crossover_params=dict(self._operators.crossover_params),
            mutation_method=mut,
            mutation_params=dict(self._operators.mutation_params),
            replacement_method=rep,
            n_replace=self._operators.n_replace,
            elitism_count=self._operators.elitism_count,
            crowding_factor=self._operators.crowding_factor,
        )

        self._notify()

    def set_population_size(self, size: int) -> None:
        self._genetic = GeneticParameters(
            population_size=size,
            chromosome_type=self._genetic.chromosome_type,
            p_crossover=self._genetic.p_crossover,
            p_mutation=self._genetic.p_mutation,
            seed=self._genetic.seed,
        )
        self._notify()

    def set_probabilities(self, p_crossover: float, p_mutation: float) -> None:
        self._genetic = GeneticParameters(
            population_size=self._genetic.population_size,
            chromosome_type=self._genetic.chromosome_type,
            p_crossover=p_crossover,
            p_mutation=p_mutation,
            seed=self._genetic.seed,
        )
        self._notify()

    def set_seed(self, seed: int | None) -> None:
        self._genetic = GeneticParameters(
            population_size=self._genetic.population_size,
            chromosome_type=self._genetic.chromosome_type,
            p_crossover=self._genetic.p_crossover,
            p_mutation=self._genetic.p_mutation,
            seed=seed,
        )
        self._notify()

    def set_stopping_criteria(
        self,
        max_generations: int = DEFAULT_MAX_GENERATIONS,
        stagnation_enabled: bool = False,
        stagnation_generations: int = DEFAULT_STAGNATION_GENERATIONS,
        stagnation_epsilon: float = DEFAULT_STAGNATION_EPSILON,
    ) -> None:
        self._stopping = StoppingCriteria(
            max_generations=max_generations,
            stagnation_enabled=stagnation_enabled,
            stagnation_generations=stagnation_generations,
            stagnation_epsilon=stagnation_epsilon,
        )
        self._notify()

    def set_selection_method(
        self, method: SelectionMethod, params: dict[str, Any] | None = None
    ) -> None:
        self._operators = OperatorConfig(
            selection_method=method,
            selection_params=params
            if params is not None
            else dict(self._operators.selection_params),
            crossover_method=self._operators.crossover_method,
            crossover_params=self._operators.crossover_params,
            mutation_method=self._operators.mutation_method,
            mutation_params=self._operators.mutation_params,
            replacement_method=self._operators.replacement_method,
            n_replace=self._operators.n_replace,
            elitism_count=self._operators.elitism_count,
            crowding_factor=self._operators.crowding_factor,
        )
        self._notify()

    def set_crossover_method(
        self, method: CrossoverMethod, params: dict[str, Any] | None = None
    ) -> None:
        self._operators = OperatorConfig(
            selection_method=self._operators.selection_method,
            selection_params=self._operators.selection_params,
            crossover_method=method,
            crossover_params=params
            if params is not None
            else dict(self._operators.crossover_params),
            mutation_method=self._operators.mutation_method,
            mutation_params=self._operators.mutation_params,
            replacement_method=self._operators.replacement_method,
            n_replace=self._operators.n_replace,
            elitism_count=self._operators.elitism_count,
            crowding_factor=self._operators.crowding_factor,
        )
        self._notify()

    def set_mutation_method(
        self, method: MutationMethod, params: dict[str, Any] | None = None
    ) -> None:
        self._operators = OperatorConfig(
            selection_method=self._operators.selection_method,
            selection_params=self._operators.selection_params,
            crossover_method=self._operators.crossover_method,
            crossover_params=self._operators.crossover_params,
            mutation_method=method,
            mutation_params=params if params is not None else dict(self._operators.mutation_params),
            replacement_method=self._operators.replacement_method,
            n_replace=self._operators.n_replace,
            elitism_count=self._operators.elitism_count,
            crowding_factor=self._operators.crowding_factor,
        )
        self._notify()

    def set_replacement_method(
        self,
        method: ReplacementMethod,
        elitism_count: int | None = None,
        n_replace: int | None = None,
        crowding_factor: int | None = None,
    ) -> None:
        self._operators = OperatorConfig(
            selection_method=self._operators.selection_method,
            selection_params=self._operators.selection_params,
            crossover_method=self._operators.crossover_method,
            crossover_params=self._operators.crossover_params,
            mutation_method=self._operators.mutation_method,
            mutation_params=self._operators.mutation_params,
            replacement_method=method,
            n_replace=n_replace,
            elitism_count=elitism_count,
            crowding_factor=crowding_factor,
        )
        self._notify()

    def set_all(
        self,
        genetic: GeneticParameters,
        operators: OperatorConfig,
        stopping: StoppingCriteria,
    ) -> None:
        """Reemplaza la configuración completa (usado en importación)."""
        self._genetic = genetic
        self._operators = operators
        self._stopping = stopping
        self._notify()
