"""Codecs para representaciones cromosómicas (RF-11, AMB-003, AMB-006, SUP-011).

Soporta las 4 representaciones:
  - BINARY: cromosomas en bits {0, 1} empaquetados por dimensión.
  - GRAY: código Gray convertido a binario estándar antes del mapeo fenotípico.
  - REAL: genotipo continuo en float64 acotado por los dominios.
  - PERMUTATION: genotipo discreto [0..N-1] sin repetidos (AMB-006).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from plataforma_ag.domain import DTYPE
from plataforma_ag.domain.bounds import bounds_to_arrays
from plataforma_ag.domain.config import VariableBound
from plataforma_ag.domain.enums import ChromosomeType, RedundantSpaceStrategy
from plataforma_ag.domain.errors import ValidationError

FloatArray = NDArray[np.float64]
Genotype = NDArray[Any]


@runtime_checkable
class ChromosomeCodec(Protocol):
    """Protocolo base para generadores y decodificadores de cromosomas."""

    chromosome_type: ChromosomeType
    bounds: tuple[VariableBound, ...]
    dimensions: int

    def random_genotype(self, rng: np.random.Generator, size: int) -> Genotype:
        """Genera una matriz de `size` individuos genotípicos aleatorios válidos."""
        ...

    def decode(self, genotypes: Genotype) -> FloatArray:
        """Decodifica la matriz de genotipos a una matriz de fenotipos (size, dimensions)."""
        ...


# --- Codec Binario Estándar --------------------------------------------------


class BinaryCodec:
    """Representación binaria estándar con mapeo fenotípico RF-06."""

    def __init__(
        self,
        bounds: Sequence[VariableBound],
        strategy: RedundantSpaceStrategy = RedundantSpaceStrategy.SATURATION,
    ) -> None:
        self.bounds = tuple(bounds)
        self.dimensions = len(self.bounds)
        self.strategy = strategy
        self.chromosome_type = ChromosomeType.BINARY

        self._lower, self._upper, self._bits = bounds_to_arrays(self.bounds)
        self._total_bits = int(np.sum(self._bits))
        self._offsets: list[tuple[int, int]] = []
        curr = 0
        for m in self._bits:
            self._offsets.append((curr, curr + int(m)))
            curr += int(m)

    @property
    def total_bits(self) -> int:
        return self._total_bits

    def random_genotype(self, rng: np.random.Generator, size: int) -> NDArray[np.uint8]:
        return rng.integers(0, 2, size=(size, self._total_bits), dtype=np.uint8)

    def decode(self, genotypes: NDArray[np.uint8]) -> FloatArray:
        pop_size = genotypes.shape[0]
        phenotypes = np.empty((pop_size, self.dimensions), dtype=DTYPE)

        for dim_idx, (start, end) in enumerate(self._offsets):
            m_bits = int(self._bits[dim_idx])
            denom = (1 << m_bits) - 1
            x_min = float(self._lower[dim_idx])
            x_max = float(self._upper[dim_idx])
            rng_val = x_max - x_min

            slice_bits = genotypes[:, start:end]
            # Convertir cada fila a entero decimal exacto
            for ind_idx in range(pop_size):
                row = slice_bits[ind_idx]
                d = 0
                for b in row:
                    d = (d << 1) | int(b)
                ratio = float(d / denom) if denom > 0 else 0.0
                phenotypes[ind_idx, dim_idx] = x_min + ratio * rng_val

        # Garantizar x_min <= x <= x_max ante redondeos IEEE 754 (RF-06)
        return np.clip(phenotypes, self._lower, self._upper)


# --- Codec Gray --------------------------------------------------------------


class GrayCodec(BinaryCodec):
    """Representación código Gray (RF-11, SUP-011).

    Convierte el genotipo Gray a binario estándar antes de aplicar la decodificación.
    """

    def __init__(
        self,
        bounds: Sequence[VariableBound],
        strategy: RedundantSpaceStrategy = RedundantSpaceStrategy.SATURATION,
    ) -> None:
        super().__init__(bounds, strategy)
        self.chromosome_type = ChromosomeType.GRAY

    def decode(self, genotypes: NDArray[np.uint8]) -> FloatArray:
        pop_size = genotypes.shape[0]
        phenotypes = np.empty((pop_size, self.dimensions), dtype=DTYPE)

        for dim_idx, (start, end) in enumerate(self._offsets):
            m_bits = int(self._bits[dim_idx])
            denom = (1 << m_bits) - 1
            x_min = float(self._lower[dim_idx])
            x_max = float(self._upper[dim_idx])
            rng_val = x_max - x_min

            gray_slice = genotypes[:, start:end]
            # Conversión vectorizada Gray -> Binario estándar: b_j = b_{j-1} ^ g_j
            binary_slice = np.bitwise_xor.accumulate(gray_slice, axis=1)

            for ind_idx in range(pop_size):
                row = binary_slice[ind_idx]
                d = 0
                for b in row:
                    d = (d << 1) | int(b)
                ratio = float(d / denom) if denom > 0 else 0.0
                phenotypes[ind_idx, dim_idx] = x_min + ratio * rng_val

        return np.clip(phenotypes, self._lower, self._upper)


# --- Codec Real --------------------------------------------------------------


class RealCodec:
    """Representación continua (parámetros reales) en float64."""

    def __init__(
        self,
        bounds: Sequence[VariableBound],
        strategy: RedundantSpaceStrategy = RedundantSpaceStrategy.SATURATION,
    ) -> None:
        self.bounds = tuple(bounds)
        self.dimensions = len(self.bounds)
        self.strategy = strategy
        self.chromosome_type = ChromosomeType.REAL
        self._lower, self._upper, _ = bounds_to_arrays(self.bounds)

    def random_genotype(self, rng: np.random.Generator, size: int) -> FloatArray:
        return rng.uniform(self._lower, self._upper, size=(size, self.dimensions)).astype(DTYPE)

    def decode(self, genotypes: FloatArray) -> FloatArray:
        # Los genotipos ya son valores en el espacio fenotípico; se asegura factibilidad (clip)
        return np.clip(genotypes, self._lower, self._upper)


# --- Codec Permutaciones -----------------------------------------------------


class PermutationCodec:
    """Representación permutaciones discretas de tamaño N (RF-11, AMB-006).

    Cada individuo es una permutación de los enteros 0..N-1 sin elementos repetidos.
    El fenotipo mapea el orden a los valores ordenados de los dominios por dimensión.
    """

    def __init__(
        self,
        bounds: Sequence[VariableBound],
        strategy: RedundantSpaceStrategy = RedundantSpaceStrategy.SATURATION,
    ) -> None:
        self.bounds = tuple(bounds)
        self.dimensions = len(self.bounds)
        self.strategy = strategy
        self.chromosome_type = ChromosomeType.PERMUTATION
        self._lower, self._upper, _ = bounds_to_arrays(self.bounds)

        # Vector de referencia ordenado dentro de los dominios (AMB-006)
        if self.dimensions == 1:
            self._reference_values = (self._lower + self._upper) / 2.0
        else:
            steps = np.linspace(0.0, 1.0, self.dimensions, dtype=DTYPE)
            self._reference_values = self._lower + steps * (self._upper - self._lower)

    def random_genotype(self, rng: np.random.Generator, size: int) -> NDArray[np.intp]:
        pop = np.empty((size, self.dimensions), dtype=np.intp)
        for i in range(size):
            pop[i] = rng.permutation(self.dimensions)
        return pop

    def decode(self, genotypes: NDArray[np.intp]) -> FloatArray:
        # Cada dimensión recibe el valor de referencia correspondiente según la permutación
        pop_size = genotypes.shape[0]
        phenotypes = np.empty((pop_size, self.dimensions), dtype=DTYPE)
        for i in range(pop_size):
            phenotypes[i] = self._reference_values[genotypes[i]]
        return phenotypes


# --- Fábrica de Codecs -------------------------------------------------------


def create_codec(
    chromosome_type: ChromosomeType,
    bounds: Sequence[VariableBound],
    strategy: RedundantSpaceStrategy = RedundantSpaceStrategy.SATURATION,
) -> ChromosomeCodec:
    """Crea la instancia de codec adecuada para la representación especificada."""
    if chromosome_type is ChromosomeType.BINARY:
        return BinaryCodec(bounds, strategy)
    if chromosome_type is ChromosomeType.GRAY:
        return GrayCodec(bounds, strategy)
    if chromosome_type is ChromosomeType.REAL:
        return RealCodec(bounds, strategy)
    if chromosome_type is ChromosomeType.PERMUTATION:
        return PermutationCodec(bounds, strategy)
    raise ValidationError(f"Tipo de cromosoma no soportado: {chromosome_type}")
