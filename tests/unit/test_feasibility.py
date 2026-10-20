"""TASK-029 — Pruebas de factibilidad con m=47 bits y espacio redundante (RF-06)."""

from __future__ import annotations

import numpy as np

from plataforma_ag.core.encoding.codecs import BinaryCodec, GrayCodec
from plataforma_ag.core.encoding.feasibility import (
    apply_feasibility_policy,
    decode_binary_dimension,
)
from plataforma_ag.domain.config import VariableBound
from plataforma_ag.domain.enums import RedundantSpaceStrategy


def test_factibilidad_m47_en_0_10_aleatorios_y_extremos() -> None:
    # RF-06 Escenario 1 (prueba obligatoria de la matriz ERS §5):
    # Dado m=47 y dominio [0, 10], ningún fenotipo es < 0.0 ni > 10.0
    m_bits = 47
    x_min, x_max = 0.0, 10.0
    bounds = (VariableBound(variable="x1", min=x_min, max=x_max, bits=m_bits),)
    codec = BinaryCodec(bounds)

    rng = np.random.default_rng(2026)
    # Generar 10000 individuos aleatorios
    pop_size = 10000
    genotypes = codec.random_genotype(rng, size=pop_size)

    phenotypes = codec.decode(genotypes)
    assert np.all(phenotypes >= 0.0), "Ningún fenotipo puede ser < 0.0"
    assert np.all(phenotypes <= 10.0), "Ningún fenotipo puede ser > 10.0"

    # Extremo todo ceros
    all_zeros = np.zeros((1, m_bits), dtype=np.uint8)
    pheno_zeros = codec.decode(all_zeros)
    assert pheno_zeros[0, 0] == 0.0

    # Extremo todo unos
    all_ones = np.ones((1, m_bits), dtype=np.uint8)
    pheno_ones = codec.decode(all_ones)
    assert pheno_ones[0, 0] == 10.0


def test_factibilidad_gray_m47_en_0_10() -> None:
    # Misma prueba para GrayCodec con m=47
    m_bits = 47
    bounds = (VariableBound(variable="x1", min=0.0, max=10.0, bits=m_bits),)
    codec = GrayCodec(bounds)

    rng = np.random.default_rng(999)
    genotypes = codec.random_genotype(rng, size=2000)
    phenotypes = codec.decode(genotypes)

    assert np.all(phenotypes >= 0.0)
    assert np.all(phenotypes <= 10.0)


def test_factibilidad_m64_limite_maximo() -> None:
    # AMB-002: límite superior de 64 bits
    m_bits = 64
    bounds = (VariableBound(variable="x1", min=-100.0, max=100.0, bits=m_bits),)
    codec = BinaryCodec(bounds)

    all_zeros = np.zeros((1, m_bits), dtype=np.uint8)
    all_ones = np.ones((1, m_bits), dtype=np.uint8)

    assert codec.decode(all_zeros)[0, 0] == -100.0
    assert codec.decode(all_ones)[0, 0] == 100.0


def test_decode_binary_dimension_division_exacta() -> None:
    d_ints = [0, (1 << 47) - 1]
    res = decode_binary_dimension(d_ints, m_bits=47, x_min=0.0, x_max=10.0)
    assert res[0] == 0.0
    assert res[1] == 10.0


def test_politica_de_factibilidad_saturacion() -> None:
    bounds = (
        VariableBound(variable="x1", min=-5.0, max=5.0),
        VariableBound(variable="x2", min=0.0, max=10.0),
    )
    # Valores desbordados
    unbounded = np.array(
        [
            [-10.0, 15.0],
            [20.0, -5.0],
            [0.0, 5.0],
        ]
    )
    clipped = apply_feasibility_policy(unbounded, bounds, RedundantSpaceStrategy.SATURATION)
    np.testing.assert_array_equal(
        clipped,
        [
            [-5.0, 10.0],
            [5.0, 0.0],
            [0.0, 5.0],
        ],
    )
