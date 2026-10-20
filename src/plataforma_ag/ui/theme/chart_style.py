"""Sincronización de estilo entre ttkbootstrap y matplotlib (TASK-055, RF-21).

Aplica la paleta semántica del tema activo a la figura, ejes, cuadrícula,
leyenda y curvas del gráfico de convergencia.
"""

from __future__ import annotations

import matplotlib.axes
import matplotlib.figure
import ttkbootstrap as ttk


def apply_chart_theme(
    figure: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    style: ttk.Style | None = None,
) -> dict[str, str]:
    """Aplica los colores semánticos de ttkbootstrap a una figura de matplotlib.

    Retorna un diccionario con los colores aplicados para las curvas.
    """
    if style is None:
        style = ttk.Style()

    colors = style.colors
    bg_color = getattr(colors, "bg", "#ffffff")
    fg_color = getattr(colors, "fg", "#222222")
    input_bg = getattr(colors, "inputbg", bg_color)
    border_color = getattr(colors, "border", "#cccccc")

    primary_color = getattr(colors, "primary", "#007bff")
    info_color = getattr(colors, "info", "#17a2b8")
    success_color = getattr(colors, "success", "#28a745")

    # Fondo de figura y ejes
    figure.patch.set_facecolor(bg_color)
    ax.set_facecolor(input_bg)

    # Títulos y etiquetas de ejes
    ax.title.set_color(fg_color)
    ax.xaxis.label.set_color(fg_color)
    ax.yaxis.label.set_color(fg_color)

    # Ticks
    ax.tick_params(axis="x", colors=fg_color)
    ax.tick_params(axis="y", colors=fg_color)

    # Bordes (spines)
    for spine in ax.spines.values():
        spine.set_color(border_color)

    # Cuadrícula
    ax.grid(True, linestyle="--", alpha=0.35, color=fg_color)

    # Leyenda si existe
    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_facecolor(input_bg)
        legend.get_frame().set_edgecolor(border_color)
        for text in legend.get_texts():
            text.set_color(fg_color)

    return {
        "best": primary_color,
        "mean": info_color,
        "elite": success_color,
        "fg": fg_color,
        "bg": bg_color,
    }
