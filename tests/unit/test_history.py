"""Pruebas unitarias de RunHistoryStore, series y RunLogger (RF-19, RF-20, RF-21)."""

from __future__ import annotations

import math

from plataforma_ag.domain.enums import LogLevel, ObjectiveSense, StopReason
from plataforma_ag.domain.run import GenerationRecord, Individual
from plataforma_ag.services.history import RunHistoryStore
from plataforma_ag.services.run_logger import RunLogger, format_timestamp


def _generate_dummy_history(n_generations: int = 100, pop_size: int = 100) -> RunHistoryStore:
    store = RunHistoryStore(sense=ObjectiveSense.MINIMIZE)

    for g in range(n_generations):
        inds = []
        for i in range(pop_size):
            fit = 100.0 / (g + 1) + (i * 0.1)
            inds.append(
                Individual(
                    individual_id=i,
                    genotype=(0, 1) * 8,
                    phenotype=(float(i), float(g)),
                    fitness=fit,
                    is_elite=(i == 0),
                )
            )
        rec = GenerationRecord(
            generation=g,
            best_fitness=inds[0].fitness,
            mean_fitness=float(sum(x.fitness for x in inds) / len(inds)),
            elite_fitness_cumulative=inds[0].fitness,
            individuals=tuple(inds),
        )
        store.add_generation(rec)

    return store


def test_paginacion_de_individuos_en_generacion_especifica() -> None:
    # Criterio de aceptación 1: Dado 200 generaciones, consultar la 57 con página de 25
    store = _generate_dummy_history(n_generations=100, pop_size=100)
    page_items, total, total_pages = store.get_individuals_page(generation=57, page=1, page_size=25)

    assert total == 100
    assert total_pages == 4
    assert len(page_items) == 25
    assert page_items[0].individual_id == 0
    assert page_items[-1].individual_id == 24

    # Segunda página
    p2_items, _, _ = store.get_individuals_page(generation=57, page=2, page_size=25)
    assert len(p2_items) == 25
    assert p2_items[0].individual_id == 25
    assert p2_items[-1].individual_id == 49


def test_consulta_de_generacion_inexistente_devuelve_vacio() -> None:
    # Criterio de aceptación 2: Consulta de generación inexistente devuelve lista vacía
    store = _generate_dummy_history(n_generations=10)
    page_items, total, total_pages = store.get_individuals_page(generation=999)
    assert page_items == []
    assert total == 0
    assert total_pages == 0


def test_series_de_convergencia_monotonia_del_elite() -> None:
    # Criterio de aceptación: Curva élite es monótona según el sentido de optimización
    store = RunHistoryStore(sense=ObjectiveSense.MINIMIZE)

    # Simular una corrida donde el mejor local sube y baja
    best_vals = [10.0, 8.5, 9.0, 7.0, 7.5, 5.0, 5.5]
    for g, b in enumerate(best_vals):
        ind = Individual(individual_id=0, genotype=(), phenotype=(), fitness=b, is_elite=True)
        rec = GenerationRecord(
            generation=g,
            best_fitness=b,
            mean_fitness=b + 2.0,
            elite_fitness_cumulative=b,
            individuals=(ind,),
        )
        store.add_generation(rec)

    series = store.get_convergence_series()
    assert len(series.generations) == len(best_vals)
    assert series.best_fitness == tuple(best_vals)

    # En minimización, el élite acumulado no puede crecer: e_t <= e_{t-1}
    elite = series.cumulative_elite
    for i in range(1, len(elite)):
        assert elite[i] <= elite[i - 1], (
            f"Élite no monótono en índice {i}: {elite[i]} > {elite[i - 1]}"
        )

    assert elite[-1] == 5.0


def test_series_con_valores_no_finitos_no_rompen_calculo() -> None:
    # Criterio de aceptación: fitness infinito no rompe el cálculo
    store = RunHistoryStore(sense=ObjectiveSense.MINIMIZE)
    ind = Individual(
        individual_id=0, genotype=(), phenotype=(), fitness=float("inf"), is_elite=True
    )
    rec = GenerationRecord(
        generation=0,
        best_fitness=float("inf"),
        mean_fitness=float("nan"),
        elite_fitness_cumulative=float("inf"),
        individuals=(ind,),
    )
    store.add_generation(rec)

    series = store.get_convergence_series()
    assert len(series.generations) == 1
    assert math.isinf(series.best_fitness[0])
    assert math.isnan(series.mean_fitness[0])


def test_run_logger_formato_y_marcas_de_tiempo() -> None:
    # RF-19: logger con timestamp en milisegundos y niveles INFO, WARN, ERROR
    logger = RunLogger(max_entries=10)
    ind = Individual(individual_id=0, genotype=(), phenotype=(), fitness=3.14159, is_elite=True)
    rec = GenerationRecord(
        generation=1,
        best_fitness=3.14159,
        mean_fitness=4.5,
        elite_fitness_cumulative=3.14159,
        individuals=(ind,),
    )

    ev_gen = logger.log_generation(rec, cumulative_elite=2.71828)
    assert ev_gen.level is LogLevel.INFO
    assert "Gen 001" in ev_gen.message
    assert "Mejor local: 3.14159" in ev_gen.message
    assert "Mejor global: 2.71828" in ev_gen.message

    ev_stop = logger.log_stop((StopReason.MAX_GENERATIONS,), generation=100)
    assert ev_stop.level is LogLevel.INFO
    assert "MAX_GENERATIONS" in ev_stop.message

    ev_warn = logger.log_warning("División por cero interceptada", generation=2)
    assert ev_warn.level is LogLevel.WARN

    entries = logger.get_entries()
    assert len(entries) == 3

    ts_str = format_timestamp(entries[0].occurred_at)
    assert ts_str.startswith("[") and ts_str.endswith("]")
    assert "." in ts_str  # Milisegundos
