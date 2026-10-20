"""TASK-031 — Pruebas de esquemas de reemplazo y elitismo (RF-16, AMB-005, AMB-012)."""

from __future__ import annotations

import numpy as np
import pytest

from plataforma_ag.core.operators.base import Population, ReplacementContext
from plataforma_ag.core.operators.replacement import (
    CrowdingReplacement,
    ElitismReplacement,
    GenerationalReplacement,
    SteadyStateReplacement,
)
from plataforma_ag.domain.errors import ValidationError


def test_elitismo_e2_conserva_dos_mejores_identicos() -> None:
    # RF-16 Escenario 1: Dado E=2, cuando se observa la generación t+1,
    # entonces los 2 mejores de t permanecen idénticos (prueba obligatoria ERS §5).
    pop_genotypes = np.array(
        [
            [1, 1],  # aptitud = 100.0 (mejor 1)
            [2, 2],  # aptitud = 90.0  (mejor 2)
            [3, 3],  # aptitud = 50.0
            [4, 4],  # aptitud = 10.0
        ]
    )
    pop_aptitude = np.array([100.0, 90.0, 50.0, 10.0])
    pop = Population(genotypes=pop_genotypes, aptitude=pop_aptitude)

    # Descendencia nueva
    off_genotypes = np.array(
        [
            [9, 9],
            [8, 8],
            [7, 7],
            [6, 6],
        ]
    )
    off_aptitude = np.array([80.0, 70.0, 60.0, 40.0])
    offspring = Population(genotypes=off_genotypes, aptitude=off_aptitude)

    rng = np.random.default_rng(42)
    ctx = ReplacementContext(generation=1, rng=rng, params={"elitism_count": 2})

    op = ElitismReplacement()
    next_pop = op.apply(pop, offspring, ctx)

    assert len(next_pop) == 4
    # Los 2 primeros deben ser los 2 mejores de la generación anterior (idénticos)
    np.testing.assert_array_equal(next_pop.genotypes[0], [1, 1])
    np.testing.assert_array_equal(next_pop.genotypes[1], [2, 2])
    assert next_pop.aptitude[0] == 100.0
    assert next_pop.aptitude[1] == 90.0


def test_estado_estable_reemplaza_exactamente_n_peores() -> None:
    # Estado estable con n=2: reemplaza a los 2 peores de pop
    pop_genotypes = np.array(
        [
            [1, 1],  # apt 100
            [2, 2],  # apt 80
            [3, 3],  # apt 30 (peor 2)
            [4, 4],  # apt 10 (peor 1)
        ]
    )
    pop_aptitude = np.array([100.0, 80.0, 30.0, 10.0])
    pop = Population(genotypes=pop_genotypes, aptitude=pop_aptitude)

    off_genotypes = np.array(
        [
            [9, 9],  # apt 95 (mejor descendiente)
            [8, 8],  # apt 85 (segundo mejor)
        ]
    )
    off_aptitude = np.array([95.0, 85.0])
    offspring = Population(genotypes=off_genotypes, aptitude=off_aptitude)

    ctx = ReplacementContext(generation=1, rng=np.random.default_rng(0), params={"n_replace": 2})
    op = SteadyStateReplacement()
    next_pop = op.apply(pop, offspring, ctx)

    assert len(next_pop) == 4
    # Los 2 peores ([3,3] y [4,4]) deben haber sido sustituidos por [9,9] y [8,8]
    genotypes_list = [list(row) for row in next_pop.genotypes]
    assert [1, 1] in genotypes_list
    assert [2, 2] in genotypes_list
    assert [9, 9] in genotypes_list
    assert [8, 8] in genotypes_list
    assert [4, 4] not in genotypes_list


def test_estado_estable_n_invalido_lanza_validation_error() -> None:
    # RF-16 Escenario 2: n >= Np se rechaza
    pop = Population(genotypes=np.zeros((10, 2)), aptitude=np.zeros(10))
    offspring = Population(genotypes=np.zeros((10, 2)), aptitude=np.zeros(10))
    ctx = ReplacementContext(generation=1, rng=np.random.default_rng(0), params={"n_replace": 10})

    with pytest.raises(ValidationError, match="n_replace debe ser < Np"):
        SteadyStateReplacement().apply(pop, offspring, ctx)


def test_crowding_solo_reemplaza_si_es_mejor() -> None:
    # AMB-012, SUP-013: el descendiente solo reemplaza al más cercano si tiene mejor aptitud
    pop_genotypes = np.array(
        [
            [0.0, 0.0],  # apt 50
            [10.0, 10.0],  # apt 50
        ]
    )
    pop_aptitude = np.array([50.0, 50.0])
    pop = Population(genotypes=pop_genotypes, aptitude=pop_aptitude)

    # Descendiente muy cercano a [0, 0] pero con PEOR aptitud (40 < 50)
    worse_offspring = Population(
        genotypes=np.array([[0.1, 0.1]]),
        aptitude=np.array([40.0]),
    )
    ctx = ReplacementContext(
        generation=1, rng=np.random.default_rng(0), params={"crowding_factor": 2}
    )
    next_pop = CrowdingReplacement().apply(pop, worse_offspring, ctx)
    # No debe reemplazar porque no es mejor
    np.testing.assert_array_equal(next_pop.genotypes[0], [0.0, 0.0])

    # Descendiente cercano a [0, 0] con MEJOR aptitud (80 > 50)
    better_offspring = Population(
        genotypes=np.array([[0.1, 0.1]]),
        aptitude=np.array([80.0]),
    )
    next_pop2 = CrowdingReplacement().apply(pop, better_offspring, ctx)
    # Debe reemplazar a [0, 0]
    np.testing.assert_array_equal(next_pop2.genotypes[0], [0.1, 0.1])
    assert next_pop2.aptitude[0] == 80.0


def test_reemplazo_generacional_puro() -> None:
    pop = Population(genotypes=np.ones((4, 2)), aptitude=np.ones(4))
    offspring = Population(genotypes=np.full((4, 2), 2), aptitude=np.full(4, 2))
    ctx = ReplacementContext(generation=1, rng=np.random.default_rng(0), params={})
    next_pop = GenerationalReplacement().apply(pop, offspring, ctx)
    np.testing.assert_array_equal(next_pop.genotypes, offspring.genotypes)
