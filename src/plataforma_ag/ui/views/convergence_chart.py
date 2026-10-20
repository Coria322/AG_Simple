"""Gráfico de evolución y convergencia con matplotlib integrado (RF-21, TASK-051).

Visualiza en tiempo real las tres curvas de fitness (mejor local, promedio y
élite histórico acumulado) con zoom, pan, barra de herramientas y sincronización de temas.
"""

from __future__ import annotations

import io
import time
import tkinter as tk

import matplotlib

matplotlib.use("TkAgg")
import ttkbootstrap as ttk
from matplotlib.backends.backend_tkagg import (  # type: ignore[attr-defined]
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from plataforma_ag.domain.run import GenerationRecord
from plataforma_ag.ui.theme.chart_style import apply_chart_theme


class ConvergenceChart(ttk.Frame):
    """Componente visual interactivo de convergencia con matplotlib."""

    def __init__(self, master: tk.Widget, padding: int = 6) -> None:
        super().__init__(master, padding=padding)

        self._generations: list[int] = []
        self._best_history: list[float] = []
        self._mean_history: list[float] = []
        self._elite_history: list[float] = []

        self._last_draw_time: float = 0.0
        self._draw_interval: float = 1.0 / 30.0  # Máximo 30 FPS (RNF-01)

        # Figura y ejes
        self._figure = Figure(figsize=(6, 4), dpi=100)
        self._ax = self._figure.add_subplot(111)

        # Crear las tres líneas reutilizables
        self._line_best = Line2D(
            [], [], color="#007bff", linestyle="--", linewidth=1.5, label="Mejor individuo local"
        )
        self._line_mean = Line2D(
            [], [], color="#17a2b8", linestyle=":", linewidth=1.5, label="Promedio de población"
        )
        self._line_elite = Line2D(
            [], [], color="#28a745", linestyle="-", linewidth=2.2, label="Élite histórico acumulado"
        )

        self._ax.add_line(self._line_best)
        self._ax.add_line(self._line_mean)
        self._ax.add_line(self._line_elite)

        self._ax.set_xlabel("Generaciones")
        self._ax.set_ylabel("Fitness f(x)")
        self._ax.legend(loc="upper right", framealpha=0.85)

        # Canvas
        self._canvas = FigureCanvasTkAgg(self._figure, master=self)
        self._canvas_widget = self._canvas.get_tk_widget()
        self._canvas_widget.pack(fill="both", expand=True)

        # Barra de herramientas estándar de matplotlib (zoom, pan, guardar)
        self._toolbar_frame = ttk.Frame(self)
        self._toolbar_frame.pack(fill="x", pady=(2, 0))
        self._toolbar = NavigationToolbar2Tk(self._canvas, self._toolbar_frame)
        self._toolbar.update()

        # Botón extra "Restablecer vista"
        btn_reset = ttk.Button(
            self._toolbar_frame,
            text="⟲ Restablecer vista",
            bootstyle="secondary-outline",
            command=self.reset_view,
            width=16,
        )
        btn_reset.pack(side="right", padx=6)

        # Aplicar estilo inicial de ttkbootstrap
        self.apply_theme()

    def apply_theme(self) -> None:
        """Sincroniza la paleta de colores de matplotlib con el tema activo."""
        colors = apply_chart_theme(self._figure, self._ax)
        self._line_best.set_color(colors["best"])
        self._line_mean.set_color(colors["mean"])
        self._line_elite.set_color(colors["elite"])
        self._canvas.draw_idle()

    def reset_view(self) -> None:
        """Restablece los límites de los ejes para encuadrar todos los datos."""
        if not self._generations:
            self._ax.set_xlim(0, 10)
            self._ax.set_ylim(0, 10)
        else:
            x_min, x_max = min(self._generations), max(self._generations)
            if x_min == x_max:
                x_max += 1
            self._ax.set_xlim(x_min, x_max)

            all_y = self._best_history + self._mean_history + self._elite_history
            valid_y = [y for y in all_y if abs(y) < 1e12]
            if valid_y:
                y_min, y_max = min(valid_y), max(valid_y)
                padding = max(0.1, (y_max - y_min) * 0.05) if y_max != y_min else 1.0
                self._ax.set_ylim(y_min - padding, y_max + padding)

        self._canvas.draw_idle()

    def clear(self) -> None:
        """Limpia los datos del gráfico."""
        self._generations.clear()
        self._best_history.clear()
        self._mean_history.clear()
        self._elite_history.clear()

        self._line_best.set_data([], [])
        self._line_mean.set_data([], [])
        self._line_elite.set_data([], [])
        self.reset_view()

    def add_generation_data(self, record: GenerationRecord, cumulative_elite: float) -> None:
        """Agrega los datos de una nueva generación y solicita redibujado con tasa controlada."""
        self._generations.append(record.generation)
        self._best_history.append(record.best_fitness)
        self._mean_history.append(record.mean_fitness)
        self._elite_history.append(cumulative_elite)

        self._line_best.set_data(self._generations, self._best_history)
        self._line_mean.set_data(self._generations, self._mean_history)
        self._line_elite.set_data(self._generations, self._elite_history)

        now = time.monotonic()
        if now - self._last_draw_time >= self._draw_interval:
            self._last_draw_time = now
            self.reset_view()

    def load_series(
        self, generations: list[int], best: list[float], mean: list[float], elite: list[float]
    ) -> None:
        """Carga una serie completa (usado en importación CSV o restauración)."""
        self._generations = list(generations)
        self._best_history = list(best)
        self._mean_history = list(mean)
        self._elite_history = list(elite)

        self._line_best.set_data(self._generations, self._best_history)
        self._line_mean.set_data(self._generations, self._mean_history)
        self._line_elite.set_data(self._generations, self._elite_history)
        self.reset_view()

    def get_chart_png_bytes(self, dpi: int = 300) -> bytes:
        """Genera y captura el gráfico en formato PNG a alta resolución para el reporte PDF."""
        buf = io.BytesIO()
        self._figure.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        return buf.getvalue()
