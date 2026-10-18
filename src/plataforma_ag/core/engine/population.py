"""Inicialización de población factible (RF-07)."""

from __future__ import annotations

import numpy as np

from plataforma_ag.core.encoding.codecs import ChromosomeCodec, Genotype
from plataforma_ag.domain.config import DEFAULT_POPULATION_SIZE
from plataforma_ag.domain.errors import ValidationError


def initialize_population(
    codec: ChromosomeCodec,
    population_size: int = DEFAULT_POPULATION_SIZE,
    rng: np.random.Generator | None = None,
) -> Genotype:
    """Genera una población inicial de tamaño `population_size` (Np ≥ 2) usando el códec dado."""
    if not isinstance(population_size, int) or isinstance(population_size, bool):
        raise ValidationError("El tamaño de población debe ser un entero.", field="population_size")
    if population_size < 2:
        raise ValidationError(
            f"El tamaño de población debe ser ≥ 2 (recibido {population_size}).",
            field="population_size",
        )

    generator = rng if rng is not None else np.random.default_rng()
    return codec.random_genotype(generator, population_size)
