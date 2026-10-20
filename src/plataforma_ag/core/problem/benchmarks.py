"""Catálogo de funciones de prueba (benchmarks) multidimensionales (RF-01).

Implementa funciones de prueba clásicas en optimización evolutiva:
  - Esfera (Sphere): continua, convexa, unimodal. Mínimo global f(0..0) = 0.
  - Rastrigin: altamente multimodal con mínimos locales regulares. Mínimo global f(0..0) = 0.
  - Rosenbrock: valle parabólico estrecho (banana valley). Mínimo global f(1..1) = 0 (N >= 2).
  - Ackley: superficie casi plana con valle central pronunciado. Mínimo global f(0..0) = 0.
  - Griewank: componentes multimodales atenuados por el producto. Mínimo global f(0..0) = 0.

Todas las funciones admiten evaluación vectorizada sobre poblaciones completas (M x N)
con tipo de datos np.float64 (PROV-NF-02, RNF-02).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from plataforma_ag.domain import DTYPE
from plataforma_ag.domain.enums import BenchmarkFunction
from plataforma_ag.domain.errors import ValidationError

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """Especificación y evaluador de una función de prueba del catálogo."""

    name: BenchmarkFunction
    label: str
    formula_latex: str
    formula_text: str
    min_dimensions: int
    default_bounds: tuple[float, float]
    global_minimum_value: float
    global_optimum_location: Callable[[int], FloatArray]
    _fn: Callable[[FloatArray], FloatArray]

    def evaluate(self, population: ArrayLike) -> FloatArray:
        """Evalúa una matriz de fenotipos (M x N) y devuelve un vector 1D (M,)."""
        pop = np.asarray(population, dtype=DTYPE)
        if pop.ndim == 1:
            pop = pop.reshape(1, -1)
        elif pop.ndim != 2:
            raise ValidationError(f"La población debe ser 1D o 2D, se recibió dimensión {pop.ndim}")

        n_dims = pop.shape[1]
        if n_dims < self.min_dimensions:
            raise ValidationError(
                f"La función '{self.label}' requiere al menos {self.min_dimensions} dimensiones "
                f"(se recibieron {n_dims})"
            )

        return self._fn(pop)


# --- Evaluadores vectorizados -----------------------------------------------


def _eval_sphere(pop: FloatArray) -> FloatArray:
    """f(x) = sum_{i=1}^N x_i^2"""
    return np.sum(np.square(pop), axis=1, dtype=DTYPE)


def _eval_rastrigin(pop: FloatArray) -> FloatArray:
    """f(x) = 10 * N + sum_{i=1}^N (x_i^2 - 10 * cos(2 * pi * x_i))"""
    n = pop.shape[1]
    term = np.square(pop) - 10.0 * np.cos(2.0 * np.pi * pop)
    return np.asarray(10.0 * n + np.sum(term, axis=1, dtype=DTYPE), dtype=DTYPE)


def _eval_rosenbrock(pop: FloatArray) -> FloatArray:
    """f(x) = sum_{i=1}^{N-1} [100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2]"""
    x_i = pop[:, :-1]
    x_next = pop[:, 1:]
    term = 100.0 * np.square(x_next - np.square(x_i)) + np.square(1.0 - x_i)
    return np.sum(term, axis=1, dtype=DTYPE)


def _eval_ackley(pop: FloatArray) -> FloatArray:
    """f(x) = -20 * exp(-0.2 * sqrt(1/N * sum x_i^2)) - exp(1/N * sum cos(2*pi*x_i)) + 20 + e"""
    n = pop.shape[1]
    sum_sq = np.sum(np.square(pop), axis=1, dtype=DTYPE)
    sum_cos = np.sum(np.cos(2.0 * np.pi * pop), axis=1, dtype=DTYPE)

    term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    result = term1 + term2 + 20.0 + float(np.e)

    # Ackley en el origen tiene tolerancia numérica para garantizar f(0) = 0.0 exacto
    zero_mask = np.all(np.abs(pop) < 1e-12, axis=1)
    if np.any(zero_mask):
        result = np.where(zero_mask, 0.0, result)

    return np.asarray(result, dtype=DTYPE)


def _eval_griewank(pop: FloatArray) -> FloatArray:
    """f(x) = 1 + 1/4000 * sum_{i=1}^N x_i^2 - prod_{i=1}^N cos(x_i / sqrt(i))"""
    n = pop.shape[1]
    sum_sq = np.sum(np.square(pop), axis=1, dtype=DTYPE)
    indices = np.arange(1, n + 1, dtype=DTYPE)
    cos_terms = np.cos(pop / np.sqrt(indices))
    prod_cos = np.prod(cos_terms, axis=1, dtype=DTYPE)

    return np.asarray(1.0 + (sum_sq / 4000.0) - prod_cos, dtype=DTYPE)


# --- Ubicaciones de óptimos globales -----------------------------------------


def _opt_zeros(n: int) -> FloatArray:
    return np.zeros(n, dtype=DTYPE)


def _opt_ones(n: int) -> FloatArray:
    return np.ones(n, dtype=DTYPE)


# --- Catálogo oficial de benchmarks -----------------------------------------

BENCHMARK_CATALOG: dict[BenchmarkFunction, BenchmarkSpec] = {
    BenchmarkFunction.SPHERE: BenchmarkSpec(
        name=BenchmarkFunction.SPHERE,
        label="Esfera (Sphere)",
        formula_latex=r"f(x) = \sum_{i=1}^N x_i^2",
        formula_text="f(x) = sum(x_i^2)",
        min_dimensions=1,
        default_bounds=(-5.12, 5.12),
        global_minimum_value=0.0,
        global_optimum_location=_opt_zeros,
        _fn=_eval_sphere,
    ),
    BenchmarkFunction.RASTRIGIN: BenchmarkSpec(
        name=BenchmarkFunction.RASTRIGIN,
        label="Rastrigin",
        formula_latex=r"f(x) = 10N + \sum_{i=1}^N [x_i^2 - 10\cos(2\pi x_i)]",
        formula_text="f(x) = 10*N + sum(x_i^2 - 10*cos(2*pi*x_i))",
        min_dimensions=1,
        default_bounds=(-5.12, 5.12),
        global_minimum_value=0.0,
        global_optimum_location=_opt_zeros,
        _fn=_eval_rastrigin,
    ),
    BenchmarkFunction.ROSENBROCK: BenchmarkSpec(
        name=BenchmarkFunction.ROSENBROCK,
        label="Rosenbrock",
        formula_latex=r"f(x) = \sum_{i=1}^{N-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]",
        formula_text="f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)",
        min_dimensions=2,
        default_bounds=(-2.048, 2.048),
        global_minimum_value=0.0,
        global_optimum_location=_opt_ones,
        _fn=_eval_rosenbrock,
    ),
    BenchmarkFunction.ACKLEY: BenchmarkSpec(
        name=BenchmarkFunction.ACKLEY,
        label="Ackley",
        formula_latex=(
            r"f(x) = -20\exp\left(-0.2\sqrt{\frac{1}{N}\sum x_i^2}\right) "
            r"- \exp\left(\frac{1}{N}\sum \cos(2\pi x_i)\right) + 20 + e"
        ),
        formula_text="f(x) = -20*exp(-0.2*sqrt(sum(x_i^2)/N)) - exp(sum(cos(2*pi*x_i))/N) + 20 + e",
        min_dimensions=1,
        default_bounds=(-32.768, 32.768),
        global_minimum_value=0.0,
        global_optimum_location=_opt_zeros,
        _fn=_eval_ackley,
    ),
    BenchmarkFunction.GRIEWANK: BenchmarkSpec(
        name=BenchmarkFunction.GRIEWANK,
        label="Griewank",
        formula_latex=(
            r"f(x) = 1 + \frac{1}{4000}\sum_{i=1}^N x_i^2 "
            r"- \prod_{i=1}^N \cos\left(\frac{x_i}{\sqrt{i}}\right)"
        ),
        formula_text="f(x) = 1 + sum(x_i^2)/4000 - prod(cos(x_i / sqrt(i)))",
        min_dimensions=1,
        default_bounds=(-600.0, 600.0),
        global_minimum_value=0.0,
        global_optimum_location=_opt_zeros,
        _fn=_eval_griewank,
    ),
}


def get_benchmark(function: BenchmarkFunction | str) -> BenchmarkSpec:
    """Devuelve la especificación del benchmark solicitado o lanza ValidationError."""
    if isinstance(function, str):
        try:
            function = BenchmarkFunction(function)
        except ValueError as err:
            raise ValidationError(f"Función benchmark desconocida: '{function}'") from err

    if function not in BENCHMARK_CATALOG:
        raise ValidationError(
            f"El benchmark '{function}' no está en el catálogo de funciones predefinidas"
        )
    return BENCHMARK_CATALOG[function]


@dataclass(frozen=True, slots=True)
class BenchmarkEvaluator:
    """Adaptador que cumple el protocolo Evaluator para usar un benchmark en ObjectiveFunction."""

    spec: BenchmarkSpec
    dimensions: int

    def __post_init__(self) -> None:
        if self.dimensions < self.spec.min_dimensions:
            raise ValidationError(
                f"La función '{self.spec.label}' requiere al menos "
                f"{self.spec.min_dimensions} dimensiones (se configuró {self.dimensions})"
            )

    def evaluate(self, population: ArrayLike) -> FloatArray:
        """Evalúa los puntos fenotípicos."""
        return self.spec.evaluate(population)
