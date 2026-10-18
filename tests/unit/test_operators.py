"""TASK-032 — Pruebas de propiedades de selección, cruce y mutación (RF-13, RF-14, RF-15)."""

from __future__ import annotations

import numpy as np

from plataforma_ag.core.operators.base import MutationContext
from plataforma_ag.core.operators.crossover import (
    ArithmeticCrossover,
    BLXAlphaCrossover,
    CXCrossover,
    FlatCrossover,
    OXCrossover,
    PMXCrossover,
)
from plataforma_ag.core.operators.mutation import (
    InversionMutation,
    NonUniformMutation,
    SwapMutation,
)
from plataforma_ag.core.operators.selection import (
    LinearRankingSelection,
    RouletteSelection,
    TournamentSelection,
)

# --- 1. Propiedades de Permutación (PMX, OX, CX, Swap, Inversion) -------------


def test_operadores_permutacion_mantienen_invariante_1000_repeticiones() -> None:
    # Criterio TASK-032: Aplicado 1000 veces, todos los resultados son permutaciones válidas 0..N-1
    rng = np.random.default_rng(2026)
    n = 8
    expected = list(range(n))

    pmx = PMXCrossover()
    ox = OXCrossover()
    cx = CXCrossover()
    swap = SwapMutation()
    inv = InversionMutation()

    ctx = MutationContext(
        generation=10,
        max_generations=100,
        p_mutation=1.0,  # forzar mutación
        lower=np.zeros(n),
        upper=np.full(n, 10.0),
    )

    for _ in range(1000):
        p1 = rng.permutation(n)
        p2 = rng.permutation(n)

        # PMX
        c1, c2 = pmx.cross(p1, p2, rng)
        assert sorted(c1) == expected and sorted(c2) == expected

        # OX
        c1, c2 = ox.cross(p1, p2, rng)
        assert sorted(c1) == expected and sorted(c2) == expected

        # CX
        c1, c2 = cx.cross(p1, p2, rng)
        assert sorted(c1) == expected and sorted(c2) == expected

        # Swap
        m_swap = swap.mutate(p1, ctx, rng)
        assert sorted(m_swap) == expected

        # Inversion
        m_inv = inv.mutate(p1, ctx, rng)
        assert sorted(m_inv) == expected


# --- 2. Cruce Aritmético (RF-14 Criterio de Aceptación) ------------------------


def test_cruce_aritmetico_p1_p2_con_a_03() -> None:
    # Dado P1=(0,0), P2=(10,10) y a=0.3, entonces C1=(7,7) y C2=(3,3)
    p1 = np.array([0.0, 0.0])
    p2 = np.array([10.0, 10.0])
    rng = np.random.default_rng(0)

    op = ArithmeticCrossover()
    c1, c2 = op.cross(p1, p2, rng, a=0.3)

    # c1 = 0.3*0 + 0.7*10 = 7.0; c2 = 0.7*0 + 0.3*10 = 3.0
    np.testing.assert_allclose(c1, [7.0, 7.0])
    np.testing.assert_allclose(c2, [3.0, 3.0])


def test_cruces_reales_respetan_intervalos() -> None:
    p1 = np.array([2.0, 8.0])
    p2 = np.array([6.0, 4.0])
    rng = np.random.default_rng(123)

    # Flat
    flat = FlatCrossover()
    c1, c2 = flat.cross(p1, p2, rng)
    assert 2.0 <= c1[0] <= 6.0 and 4.0 <= c1[1] <= 8.0
    assert 2.0 <= c2[0] <= 6.0 and 4.0 <= c2[1] <= 8.0

    # BLX-alpha
    blx = BLXAlphaCrossover()
    c1, c2 = blx.cross(p1, p2, rng, alpha=0.5)
    # Rango extendido para dim 0: [2 - 0.5*4, 6 + 0.5*4] = [0, 8]
    assert 0.0 <= c1[0] <= 8.0


# --- 3. Mutación No Uniforme Decreciente con t (RF-15) -------------------------


def test_mutacion_no_uniforme_perturbacion_decrece_con_t() -> None:
    # RF-15 Criterio de Aceptación:
    # Dado b=2.0 y Gmax=200, la perturbación media en t=1 es mayor que en t=199
    op = NonUniformMutation()
    rng = np.random.default_rng(42)
    n_samples = 2000
    ind = np.array([5.0])
    lower = np.array([0.0])
    upper = np.array([10.0])

    ctx_t1 = MutationContext(
        generation=1, max_generations=200, p_mutation=1.0, lower=lower, upper=upper
    )
    ctx_t199 = MutationContext(
        generation=199, max_generations=200, p_mutation=1.0, lower=lower, upper=upper
    )

    diffs_t1 = [abs(op.mutate(ind, ctx_t1, rng, b=2.0)[0] - 5.0) for _ in range(n_samples)]
    diffs_t199 = [abs(op.mutate(ind, ctx_t199, rng, b=2.0)[0] - 5.0) for _ in range(n_samples)]

    mean_diff_t1 = float(np.mean(diffs_t1))
    mean_diff_t199 = float(np.mean(diffs_t199))

    assert mean_diff_t1 > mean_diff_t199, (
        f"t1 ({mean_diff_t1}) debió ser mayor que t199 ({mean_diff_t199})"
    )


# --- 4. Selección por Torneo (RF-13) -------------------------------------------


def test_seleccion_torneo_k3_elige_el_mejor_de_tres() -> None:
    # RF-13 Criterio: Dado k=3, el padre elegido es el de mayor aptitud entre los 3 extraídos
    # Población con un único individuo dominante
    fitness = np.array([1.0, 1.0, 1.0, 100.0, 1.0])  # índice 3 es dominante
    rng = np.random.default_rng(10)
    torneo = TournamentSelection()

    selected = torneo.select(fitness, n=50, rng=rng, k=3)
    # Por probabilidad combinatoria (1 - (4/5)^3 ≈ 0.488), el individuo 3 debe ganar muchas veces
    assert 3 in selected
    assert np.count_nonzero(selected == 3) > 15


# --- 5. Probabilidades de Selección y Uniformidad (RF-13) ----------------------


def test_seleccion_uniforme_ante_aptitudes_iguales() -> None:
    # RF-13 Criterio: Dado aptitudes iguales, la selección es uniforme
    fitness = np.full(10, 5.0)
    rng = np.random.default_rng(0)

    roulette = RouletteSelection()
    selected = roulette.select(fitness, n=1000, rng=rng)
    counts = np.bincount(selected, minlength=10)
    # Todos deben haber sido elegidos aproximadamente 100 veces (dentro de márgenes razonables)
    assert np.all(counts > 50) and np.all(counts < 150)


def test_ranking_lineal_probabilidades_suman_uno() -> None:
    fitness = np.array([10.0, 20.0, 30.0, 40.0])
    rng = np.random.default_rng(0)
    linear = LinearRankingSelection()
    selected = linear.select(fitness, n=200, rng=rng, s=1.8)
    assert len(selected) == 200
    # Individuos con mayor fitness (índice 3) son elegidos con mayor frecuencia
    counts = np.bincount(selected, minlength=4)
    assert counts[3] > counts[0]
