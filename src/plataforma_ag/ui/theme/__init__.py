"""Submódulo de temas y estilos para la interfaz gráfica."""

from plataforma_ag.ui.theme.chart_style import apply_chart_theme
from plataforma_ag.ui.theme.theme_manager import DEFAULT_THEME, ThemeManager

__all__ = [
    "DEFAULT_THEME",
    "ThemeManager",
    "apply_chart_theme",
]
