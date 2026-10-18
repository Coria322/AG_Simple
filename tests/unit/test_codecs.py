"""TASK-008 — Pruebas de codecs cromosómicos (RF-11, AMB-003, AMB-006, SUP-011)."""

from __future__ import annotations

import numpy as np

from plataforma_ag.core.encoding.codecs import (
    BinaryCodec,
    GrayCodec,
    PermutationCodec,
    RealCodec,
    create_codec,
)
from plataforma_ag.domain.config import VariableBound
from plataforma_ag.domain.enums import ChromosomeType


def test_codec_binario_extremos_y_valores_medios() -> None:
    bounds = (
        VariableBound(variable="x1", min=0.0, max=10.0, bits=4),  # 0..15 -> 0, 10
        VariableBound(variable="x2", min=-5.0, max=5.0, bits=4),
    )
    codec = BinaryCodec(bounds)
    assert codec.total_bits == 8

    # Individuo todo ceros y todo unos
    genotypes = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype=np.uint8)

    phenotypes = codec.decode(genotypes)
    np.testing.assert_allclose(phenotypes[0], [0.0, -5.0])
    np.testing.assert_allclose(phenotypes[1], [10.0, 5.0])


def test_gray_decodifica_d5_a_entero_5() -> None:
    # RF-11 Escenario 1: genotipo Gray de D=5 decodifica al entero 5
    # En código Gray de 3 bits:
    # 0 -> 000, 1 -> 001, 2 -> 011, 3 -> 010, 4 -> 110, 5 -> 111, 6 -> 101, 7 -> 100
    # Para m=3 bits en dominio [0, 7], cada entero D coincide exactamente con su valor continuo.
    bounds = (VariableBound(variable="x1", min=0.0, max=7.0, bits=3),)
    gray_codec = GrayCodec(bounds)

    # 5 en Gray es 111 (bits: 1, 1, 1)
    genotypes = np.array([[1, 1, 1]], dtype=np.uint8)
    phenotype = gray_codec.decode(genotypes)

    # Debe ser exactamente 5.0
    assert phenotype[0, 0] == 5.0


def test_gray_todos_los_valores_coinciden_con_binario() -> None:
    # Probar exhaustivamente los 16 enteros de 4 bits
    bounds = (VariableBound(variable="x1", min=0.0, max=15.0, bits=4),)
    bin_codec = BinaryCodec(bounds)
    gray_codec = GrayCodec(bounds)

    for val in range(16):
        # Conversión decimal -> binario
        b_bits = np.array([(val >> (3 - i)) & 1 for i in range(4)], dtype=np.uint8)
        # Conversión decimal -> gray: g = v ^ (v >> 1)
        gray_val = val ^ (val >> 1)
        g_bits = np.array([(gray_val >> (3 - i)) & 1 for i in range(4)], dtype=np.uint8)

        bin_pheno = bin_codec.decode(b_bits.reshape(1, -1))
        gray_pheno = gray_codec.decode(g_bits.reshape(1, -1))

        assert bin_pheno[0, 0] == float(val)
        assert gray_pheno[0, 0] == float(val)


def test_codec_permutaciones_sin_repetidos() -> None:
    # RF-11 Escenario 2: permutación de tamaño 8 contiene 0..7 sin repetidos
    bounds = tuple(VariableBound(variable=f"x{i+1}", min=0.0, max=10.0) for i in range(8))
    codec = PermutationCodec(bounds)
    rng = np.random.default_rng(42)

    genotypes = codec.random_genotype(rng, size=10)
    assert genotypes.shape == (10, 8)

    for ind in genotypes:
        assert sorted(ind) == list(range(8))


def test_codec_real_en_rango() -> None:
    bounds = (
        VariableBound(variable="x1", min=-10.0, max=10.0),
        VariableBound(variable="x2", min=100.0, max=200.0),
    )
    codec = RealCodec(bounds)
    rng = np.random.default_rng(123)

    genotypes = codec.random_genotype(rng, size=50)
    assert genotypes.shape == (50, 2)
    assert np.all(genotypes[:, 0] >= -10.0) and np.all(genotypes[:, 0] <= 10.0)
    assert np.all(genotypes[:, 1] >= 100.0) and np.all(genotypes[:, 1] <= 200.0)

    # Decodificación respeta límites
    pheno = codec.decode(genotypes)
    np.testing.assert_array_equal(pheno, genotypes)


def test_create_codec_factory() -> None:
    bounds = (VariableBound(variable="x1", min=0.0, max=1.0),)
    assert isinstance(create_codec(ChromosomeType.BINARY, bounds), BinaryCodec)
    assert isinstance(create_codec(ChromosomeType.GRAY, bounds), GrayCodec)
    assert isinstance(create_codec(ChromosomeType.REAL, bounds), RealCodec)
    assert isinstance(create_codec(ChromosomeType.PERMUTATION, bounds), PermutationCodec)
