"""Pruebas smoke de UI: temas, barra, tabla y gráfica (TASK-059, PROV-UI-01)."""

from __future__ import annotations

import tkinter as tk

import pytest

from plataforma_ag.controllers.params_controller import ParamsController
from plataforma_ag.controllers.persistence_controller import PersistenceController
from plataforma_ag.controllers.problem_controller import ProblemController
from plataforma_ag.controllers.run_controller import RunController
from plataforma_ag.controllers.telemetry_controller import TelemetryController
from plataforma_ag.domain.enums import RunState
from plataforma_ag.domain.run import GenerationRecord, Individual, ProgressSnapshot
from plataforma_ag.ui.theme.theme_manager import ThemeManager
from plataforma_ag.ui.views.main_window import MainWindow


def test_theme_manager_temas_disponibles_sin_fallo() -> None:
    try:
        mgr = ThemeManager()
        themes = mgr.available_themes()
        assert len(themes) > 0
        assert "flatly" in themes or len(themes) >= 1
    except tk.TclError:
        pytest.skip("No hay display gráfico disponible en este entorno.")


@pytest.mark.ui_smoke
def test_main_window_ensamblado_y_actualizaciones() -> None:
    try:
        problem_ctrl = ProblemController()
        params_ctrl = ParamsController()
        telemetry_ctrl = TelemetryController()
        run_ctrl = RunController(telemetry_ctrl)
        persistence_ctrl = PersistenceController()
        theme_mgr = ThemeManager()

        win = MainWindow(
            problem_controller=problem_ctrl,
            params_controller=params_ctrl,
            run_controller=run_ctrl,
            telemetry_controller=telemetry_ctrl,
            persistence_controller=persistence_ctrl,
            theme_manager=theme_mgr,
        )

        # Probar actualización de progreso
        snap = ProgressSnapshot(
            current_generation=10, max_generations=100, percent=10.0, state=RunState.RUNNING
        )
        win.progress_view.update_progress(snap)

        # Probar actualización de estado
        win.control_bar.update_state(RunState.RUNNING)
        win.progress_view.update_state(RunState.RUNNING)

        # Probar agregar generación al gráfico y tabla
        ind = Individual(individual_id=0, genotype=(1,), phenotype=(0.5,), fitness=2.5)
        rec = GenerationRecord(
            generation=1,
            best_fitness=2.5,
            mean_fitness=3.0,
            elite_fitness_cumulative=2.5,
            individuals=(ind,),
        )
        win.chart_view.add_generation_data(rec, cumulative_elite=2.5)
        win.pop_table.update_max_generation(1)
        win.pop_table.refresh()

        # Probar exportación de imagen PNG del gráfico
        png_bytes = win.chart_view.get_chart_png_bytes(dpi=72)
        assert len(png_bytes) > 0
        assert png_bytes.startswith(b"\x89PNG")

        # Destruir ventana limpia
        win.root.destroy()
    except tk.TclError:
        pytest.skip("No hay display gráfico disponible en este entorno.")
