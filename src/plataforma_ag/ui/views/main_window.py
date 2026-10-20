"""Ventana principal integrada de la plataforma (TASK-020, UI_UX_SPEC.md).

Ensambla menús, pestañas de configuración, visualizador de convergencia,
telemetría en tiempo real, consola de bitácora y barra de control.
"""

from __future__ import annotations

import contextlib
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Any

import ttkbootstrap as ttk
from ttkbootstrap import ToastNotification
from ttkbootstrap.dialogs import Messagebox

from plataforma_ag import __version__
from plataforma_ag.controllers.params_controller import ParamsController
from plataforma_ag.controllers.persistence_controller import PersistenceController
from plataforma_ag.controllers.problem_controller import ProblemController
from plataforma_ag.controllers.run_controller import RunController
from plataforma_ag.controllers.telemetry_controller import TelemetryController
from plataforma_ag.domain.config import ExperimentConfig
from plataforma_ag.domain.enums import ChromosomeType, RunState
from plataforma_ag.domain.run import GenerationRecord, ProgressSnapshot
from plataforma_ag.domain.validation import validate_experiment_config
from plataforma_ag.ui.theme.theme_manager import DEFAULT_THEME, ThemeManager
from plataforma_ag.ui.views.control_bar import ControlBar
from plataforma_ag.ui.views.convergence_chart import ConvergenceChart
from plataforma_ag.ui.views.log_console import LogConsole
from plataforma_ag.ui.views.operators_view import OperatorsView
from plataforma_ag.ui.views.params_view import ParamsView
from plataforma_ag.ui.views.population_table import PopulationTable
from plataforma_ag.ui.views.problem_view import ProblemView
from plataforma_ag.ui.views.progress_view import ProgressView


class MainWindow:
    """Ventana principal de la Plataforma AG con arquitectura MVC."""

    def __init__(
        self,
        problem_controller: ProblemController | None = None,
        params_controller: ParamsController | None = None,
        run_controller: RunController | None = None,
        telemetry_controller: TelemetryController | None = None,
        persistence_controller: PersistenceController | None = None,
        theme_manager: ThemeManager | None = None,
    ) -> None:
        self.problem_ctrl = problem_controller or ProblemController()
        self.params_ctrl = params_controller or ParamsController()
        self.telemetry_ctrl = telemetry_controller or TelemetryController()
        self.run_ctrl = run_controller or RunController(self.telemetry_ctrl)
        self.persistence_ctrl = persistence_controller or PersistenceController()
        self.theme_mgr = theme_manager or ThemeManager(DEFAULT_THEME)

        # Garantizar que no quede un root huérfano de un Style previo
        existing_style = ttk.Style.get_instance()
        if existing_style is not None and getattr(existing_style, "master", None):
            with contextlib.suppress(Exception):
                existing_style.master.destroy()

        # Ventana principal
        self.root = ttk.Window(
            title=f"Plataforma AG — Optimización con Algoritmos Genéticos v{__version__}",
            themename=self.theme_mgr.current_theme,
            minsize=(1100, 720),
        )

        self._build_menubar()
        self._build_layout()
        self._setup_event_listeners()

        # Iniciar ciclo de sondeo de eventos a 30 FPS (33 ms)
        self.root.after(33, self._poll_tick)

    def _build_menubar(self) -> None:
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Menú Archivo
        file_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="Archivo", menu=file_menu)

        file_menu.add_command(
            label="Importar configuración (JSON)...", command=self._action_import_json
        )
        file_menu.add_command(
            label="Exportar configuración (JSON)...", command=self._action_export_json
        )
        file_menu.add_separator()
        file_menu.add_command(label="Importar telemetría (CSV)...", command=self._action_import_csv)
        file_menu.add_command(label="Exportar telemetría (CSV)...", command=self._action_export_csv)
        file_menu.add_separator()
        file_menu.add_command(
            label="Generar informe técnico (PDF)...", command=self._action_generate_pdf
        )
        file_menu.add_separator()
        file_menu.add_command(label="Salir", command=self.root.quit)

        # Menú Vista
        view_menu = tk.Menu(menubar, tearoff=False)
        menubar.add_cascade(label="Vista", menu=view_menu)

        theme_submenu = tk.Menu(view_menu, tearoff=False)
        view_menu.add_cascade(label="Tema", menu=theme_submenu)

        available_themes = self.theme_mgr.available_themes()
        self._theme_var = tk.StringVar(value=self.theme_mgr.current_theme)
        for tname in available_themes:
            theme_submenu.add_radiobutton(
                label=tname.capitalize(),
                value=tname,
                variable=self._theme_var,
                command=self._make_theme_cmd(tname),
            )

    def _make_theme_cmd(self, theme_name: str) -> Any:
        return lambda: self._change_theme(theme_name)

    def _build_layout(self) -> None:
        # Contenedor superior con barra de control y barra de progreso
        top_container = ttk.Frame(self.root, padding=(10, 8))
        top_container.pack(fill="x")

        self.control_bar = ControlBar(
            top_container,
            on_start=self._start_simulation,
            on_pause=self.run_ctrl.pause_run,
            on_resume=self.run_ctrl.resume_run,
            on_abort=self.run_ctrl.abort_run,
        )
        self.control_bar.pack(side="left")

        self.progress_view = ProgressView(top_container)
        self.progress_view.pack(side="right", fill="x", expand=True)

        # Separador horizontal
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=10, pady=(0, 4))

        # Panedwindow principal (Horizontal: Izquierda Configuración, Derecha Datos)
        main_paned = ttk.Panedwindow(self.root, orient="horizontal")
        main_paned.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Panel Izquierdo: Notebook de Configuración
        config_frame = ttk.Frame(main_paned, width=420)
        main_paned.add(config_frame, weight=1)

        self.config_notebook = ttk.Notebook(config_frame)
        self.config_notebook.pack(fill="both", expand=True)

        self.problem_view = ProblemView(
            self.config_notebook,
            controller=self.problem_ctrl,
            on_validation_changed=self._on_form_validity_changed,
        )
        self.config_notebook.add(self.problem_view, text="1. Problema")

        self.params_view = ParamsView(
            self.config_notebook,
            controller=self.params_ctrl,
            on_validation_changed=self._on_form_validity_changed,
        )
        self.config_notebook.add(self.params_view, text="2. Población")

        self.operators_view = OperatorsView(
            self.config_notebook,
            controller=self.params_ctrl,
            on_representation_changed=self._on_representation_changed,
        )
        self.config_notebook.add(self.operators_view, text="3. Operadores")

        # Panel Derecho: Gráfica y Telemetría divididos verticalmente
        right_paned = ttk.Panedwindow(main_paned, orient="vertical")
        main_paned.add(right_paned, weight=2)

        # Gráfico superior
        chart_container = ttk.Frame(right_paned)
        right_paned.add(chart_container, weight=3)

        self.chart_view = ConvergenceChart(chart_container)
        self.chart_view.pack(fill="both", expand=True)

        # Telemetría inferior
        telemetry_container = ttk.Frame(right_paned)
        right_paned.add(telemetry_container, weight=2)

        self.telemetry_notebook = ttk.Notebook(telemetry_container)
        self.telemetry_notebook.pack(fill="both", expand=True)

        self.pop_table = PopulationTable(
            self.telemetry_notebook, telemetry_controller=self.telemetry_ctrl
        )
        self.telemetry_notebook.add(self.pop_table, text="Tabla de Población")

        self.log_console = LogConsole(self.telemetry_notebook)
        self.telemetry_notebook.add(self.log_console, text="Bitácora")

    def _setup_event_listeners(self) -> None:
        self.run_ctrl.add_state_listener(self._on_run_state_changed)
        self.telemetry_ctrl.add_progress_listener(self._on_progress_update)
        self.telemetry_ctrl.add_generation_listener(self._on_generation_update)
        self.telemetry_ctrl.add_log_listener(self.log_console.add_log_event)

        self.persistence_ctrl.add_io_start_listener(self.progress_view.show_io_activity)
        self.persistence_ctrl.add_io_finish_listener(self.progress_view.hide_io_activity)

        self.theme_mgr.add_theme_listener(self._on_theme_applied)

    def _poll_tick(self) -> None:
        """Sondeo periódico de la cola de eventos del motor (cada 33 ms)."""
        self.run_ctrl.poll_events()
        self.root.after(33, self._poll_tick)

    def _start_simulation(self) -> None:
        """Compila y valida la configuración actual e inicia la simulación."""
        cfg = ExperimentConfig(
            problem=self.problem_ctrl.problem,
            genetic=self.params_ctrl.genetic,
            operators=self.params_ctrl.operators,
            stopping=self.params_ctrl.stopping,
        )

        res = validate_experiment_config(cfg)
        if res.has_errors:
            err_msg = "\n".join(f"• {k}: {v}" for k, v in res.errors.items())
            Messagebox.show_error(
                message=f"La configuración contiene errores:\n\n{err_msg}",
                title="Configuración inválida",
                parent=self.root,
            )
            return

        self.chart_view.clear()
        self.log_console.clear()
        self.run_ctrl.start_run(cfg)

    def _on_run_state_changed(self, state: RunState) -> None:
        self.control_bar.update_state(state)
        self.progress_view.update_state(state)

        # Bloquear paneles de configuración mientras corre
        is_running_or_paused = state in (RunState.RUNNING, RunState.PAUSED)
        # Deshabilitar interacción en las pestañas
        for child in (self.problem_view, self.params_view, self.operators_view):
            for w in child.winfo_children():
                with contextlib.suppress(Exception):
                    w.configure(**{"state": "disabled" if is_running_or_paused else "normal"})

    def _on_progress_update(self, snapshot: ProgressSnapshot) -> None:
        self.progress_view.update_progress(snapshot)

    def _on_generation_update(self, record: GenerationRecord, cumulative_elite: float) -> None:
        self.chart_view.add_generation_data(record, cumulative_elite)
        self.pop_table.update_max_generation(record.generation)

    def _on_form_validity_changed(self, is_valid: bool) -> None:
        self.control_bar.set_form_valid(is_valid)

    def _on_representation_changed(self, chrom: ChromosomeType) -> None:
        self.problem_view.set_chromosome_type(chrom)

    def _change_theme(self, theme_name: str) -> None:
        ok = self.theme_mgr.apply_theme(theme_name)
        if not ok:
            ToastNotification(
                title="Error de tema",
                message=f"No se pudo aplicar el tema '{theme_name}'",
                bootstyle="danger",
                duration=3000,
            ).show_toast()

    def _on_theme_applied(self, theme_name: str) -> None:
        self.chart_view.apply_theme()

    def _action_export_json(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension=".json",
            filetypes=[("Archivos JSON", "*.json"), ("Todos los archivos", "*.*")],
            title="Exportar configuración del experimento",
        )
        if not path:
            return

        cfg = ExperimentConfig(
            problem=self.problem_ctrl.problem,
            genetic=self.params_ctrl.genetic,
            operators=self.params_ctrl.operators,
            stopping=self.params_ctrl.stopping,
        )
        try:
            self.persistence_ctrl.save_config_json(path, cfg)
            ToastNotification(
                title="Configuración exportada",
                message=f"Archivo guardado en {Path(path).name}",
                bootstyle="success",
                duration=3000,
            ).show_toast()
        except Exception as ex:
            Messagebox.show_error(
                f"Error al exportar JSON: {ex}", title="Error de exportación", parent=self.root
            )

    def _action_import_json(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            filetypes=[("Archivos JSON", "*.json"), ("Todos los archivos", "*.*")],
            title="Importar configuración del experimento",
        )
        if not path:
            return

        try:
            cfg = self.persistence_ctrl.load_config_json(path)
            self.problem_ctrl.set_entire_problem(cfg.problem)
            self.params_ctrl.set_all(cfg.genetic, cfg.operators, cfg.stopping)
            ToastNotification(
                title="Configuración importada",
                message=f"Configuración cargada desde {Path(path).name}",
                bootstyle="success",
                duration=3000,
            ).show_toast()
        except Exception as ex:
            Messagebox.show_error(
                f"Error al importar JSON: {ex}", title="Error de importación", parent=self.root
            )

    def _action_export_csv(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension=".csv",
            filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")],
            title="Exportar telemetría de la simulación",
        )
        if not path:
            return

        try:
            self.persistence_ctrl.export_csv(path, self.telemetry_ctrl.history_store)
            ToastNotification(
                title="Telemetría exportada",
                message=f"Archivo CSV guardado en {Path(path).name}",
                bootstyle="success",
                duration=3000,
            ).show_toast()
        except Exception as ex:
            Messagebox.show_error(
                f"Error al exportar CSV: {ex}", title="Error de exportación", parent=self.root
            )

    def _action_import_csv(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")],
            title="Importar telemetría de corrida previa",
        )
        if not path:
            return

        try:
            store = self.persistence_ctrl.import_csv(path)
            self.telemetry_ctrl.history_store = store
            series = store.get_series()
            self.chart_view.load_series(
                generations=list(series.generations),
                best=list(series.best_fitness),
                mean=list(series.mean_fitness),
                elite=list(series.cumulative_elite),
            )
            max_g = max(series.generations) if series.generations else 0
            self.pop_table.update_max_generation(max_g)
            self.pop_table.refresh()

            ToastNotification(
                title="Telemetría importada",
                message=f"Historial cargado desde {Path(path).name} sin reejecutar.",
                bootstyle="success",
                duration=3000,
            ).show_toast()
        except Exception as ex:
            Messagebox.show_error(
                f"Error al importar CSV: {ex}", title="Error de importación", parent=self.root
            )

    def _action_generate_pdf(self) -> None:
        if len(self.telemetry_ctrl.history_store) == 0:
            Messagebox.show_warning(
                "No hay datos de ninguna corrida para generar el reporte.",
                title="Sin datos",
                parent=self.root,
            )
            return

        path = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension=".pdf",
            filetypes=[("Documentos PDF", "*.pdf"), ("Todos los archivos", "*.*")],
            title="Guardar informe técnico formal",
        )
        if not path:
            return

        chart_bytes = self.chart_view.get_chart_png_bytes(dpi=300)
        cfg = ExperimentConfig(
            problem=self.problem_ctrl.problem,
            genetic=self.params_ctrl.genetic,
            operators=self.params_ctrl.operators,
            stopping=self.params_ctrl.stopping,
        )

        def _on_success(out_path: Path) -> None:
            self.root.after(
                0,
                lambda: ToastNotification(
                    title="Informe PDF generado",
                    message=f"Reporte técnico guardado en {out_path.name}",
                    bootstyle="success",
                    duration=4000,
                ).show_toast(),
            )

        def _on_error(err: Exception) -> None:
            self.root.after(
                0,
                lambda: Messagebox.show_error(
                    f"Error al generar informe PDF: {err}",
                    title="Error de reporte",
                    parent=self.root,
                ),
            )

        self.persistence_ctrl.generate_pdf_async(
            path=path,
            config=cfg,
            history=self.telemetry_ctrl.history_store,
            chart_png_bytes=chart_bytes,
            on_success=_on_success,
            on_error=_on_error,
        )

    def run(self) -> None:
        """Inicia el bucle de eventos de la aplicación."""
        self.root.mainloop()
