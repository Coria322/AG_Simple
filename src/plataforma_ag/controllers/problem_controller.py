"""Controlador del modelo de definición del problema (RF-01..RF-06, TASK-021..023).

Gestiona la función objetivo (benchmark o expresión libre), dimensiones,
sentido de optimización y límites dinámicos por variable.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

from plataforma_ag.core.problem.benchmarks import BENCHMARK_CATALOG
from plataforma_ag.core.problem.expression import compile_expression
from plataforma_ag.domain.config import (
    DEFAULT_BITS,
    DEMO_BOUND_MAX,
    DEMO_BOUND_MIN,
    ProblemDefinition,
    VariableBound,
    variable_name,
)
from plataforma_ag.domain.enums import (
    BenchmarkFunction,
    ChromosomeType,
    ObjectiveSense,
)
from plataforma_ag.domain.errors import ExpressionError


class ProblemController:
    """Controlador que encapsula y valida la configuración del problema matemático."""

    def __init__(self, initial_problem: ProblemDefinition | None = None) -> None:
        self._problem = initial_problem or ProblemDefinition()
        self._listeners: list[Callable[[ProblemDefinition], None]] = []

    @property
    def problem(self) -> ProblemDefinition:
        return self._problem

    def add_listener(self, callback: Callable[[ProblemDefinition], None]) -> None:
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[ProblemDefinition], None]) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify(self) -> None:
        for cb in list(self._listeners):
            with contextlib.suppress(Exception):
                cb(self._problem)

    def set_benchmark(self, benchmark: BenchmarkFunction) -> None:
        """Selecciona una función de benchmark o la opción de expresión libre."""
        if self._problem.benchmark == benchmark:
            return

        spec = BENCHMARK_CATALOG.get(benchmark)
        dims = self._problem.dimensions
        if spec is not None:
            if dims < spec.min_dimensions:
                dims = spec.min_dimensions
            b_min, b_max = spec.default_bounds
            bounds = tuple(
                VariableBound(variable=variable_name(i), min=b_min, max=b_max) for i in range(dims)
            )
            expr = spec.label
        else:
            expr = self._problem.expression or "x1^2 + x2^2"
            bounds = self._problem.bounds

        self._problem = ProblemDefinition(
            benchmark=benchmark,
            dimensions=dims,
            expression=expr,
            angle_unit=self._problem.angle_unit,
            objective=self._problem.objective,
            redundant_space_strategy=self._problem.redundant_space_strategy,
            fitness_transform=self._problem.fitness_transform,
            bounds=bounds,
        )
        self._notify()

    def set_dimensions(
        self, new_dimensions: int, chromosome_type: ChromosomeType = ChromosomeType.BINARY
    ) -> None:
        """Actualiza el número de dimensiones N, preservando los límites existentes."""
        if new_dimensions < 1:
            raise ValueError("Las dimensiones deben ser al menos 1")

        curr_bounds = list(self._problem.bounds)
        if len(curr_bounds) < new_dimensions:
            # Expandir dimensiones conservando las anteriores
            for i in range(len(curr_bounds), new_dimensions):
                v_name = variable_name(i)
                curr_bounds.append(
                    VariableBound(
                        variable=v_name,
                        min=DEMO_BOUND_MIN,
                        max=DEMO_BOUND_MAX,
                        bits=DEFAULT_BITS,
                    )
                )
        elif len(curr_bounds) > new_dimensions:
            # Truncar
            curr_bounds = curr_bounds[:new_dimensions]

        self._problem = ProblemDefinition(
            benchmark=self._problem.benchmark,
            dimensions=new_dimensions,
            expression=self._problem.expression,
            angle_unit=self._problem.angle_unit,
            objective=self._problem.objective,
            redundant_space_strategy=self._problem.redundant_space_strategy,
            fitness_transform=self._problem.fitness_transform,
            bounds=tuple(curr_bounds),
        )
        self._notify()

    def set_objective_sense(self, sense: ObjectiveSense) -> None:
        """Establece el sentido de optimización (Minimizar / Maximizar)."""
        if self._problem.objective == sense:
            return

        self._problem = ProblemDefinition(
            benchmark=self._problem.benchmark,
            dimensions=self._problem.dimensions,
            expression=self._problem.expression,
            angle_unit=self._problem.angle_unit,
            objective=sense,
            redundant_space_strategy=self._problem.redundant_space_strategy,
            fitness_transform=self._problem.fitness_transform,
            bounds=self._problem.bounds,
        )
        self._notify()

    def set_expression(self, expression: str) -> None:
        """Establece la expresión matemática para el caso personalizado."""
        self._problem = ProblemDefinition(
            benchmark=self._problem.benchmark,
            dimensions=self._problem.dimensions,
            expression=expression,
            angle_unit=self._problem.angle_unit,
            objective=self._problem.objective,
            redundant_space_strategy=self._problem.redundant_space_strategy,
            fitness_transform=self._problem.fitness_transform,
            bounds=self._problem.bounds,
        )
        self._notify()

    def update_variable_bound(
        self,
        index: int,
        min_val: float,
        max_val: float,
        bits: int = DEFAULT_BITS,
    ) -> None:
        """Actualiza el dominio y resolución para la dimensión indicada."""
        if not (0 <= index < len(self._problem.bounds)):
            return

        bounds_list = list(self._problem.bounds)
        curr = bounds_list[index]
        bounds_list[index] = VariableBound(
            variable=curr.variable,
            min=min_val,
            max=max_val,
            bits=bits,
        )

        self._problem = ProblemDefinition(
            benchmark=self._problem.benchmark,
            dimensions=self._problem.dimensions,
            expression=self._problem.expression,
            angle_unit=self._problem.angle_unit,
            objective=self._problem.objective,
            redundant_space_strategy=self._problem.redundant_space_strategy,
            fitness_transform=self._problem.fitness_transform,
            bounds=tuple(bounds_list),
        )
        self._notify()

    def set_entire_problem(self, new_problem: ProblemDefinition) -> None:
        """Reemplaza la definición completa del problema (usado en importación)."""
        self._problem = new_problem
        self._notify()

    def validate_expression(self, expression: str) -> str | None:
        """Valida sintaxis y variables de una expresión matemática."""
        if not expression or not expression.strip():
            return "La expresión matemática no puede estar vacía"

        try:
            compiled = compile_expression(expression, self._problem.dimensions)
            for idx in compiled.variables_used:
                if idx > self._problem.dimensions:
                    return (
                        f"Variable 'x{idx}' no pertenece a las "
                        f"{self._problem.dimensions} dimensiones configuradas"
                    )
            return None
        except ExpressionError as ex:
            return str(ex)
        except Exception as ex:
            return f"Error en la expresión: {ex}"
