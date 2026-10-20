"""Gestor de temas de ttkbootstrap (PROV-UI-01, TASK-055).

Permite consultar los temas disponibles en tiempo de ejecución (sin hardcodear listas)
y aplicar temas dinámicamente notificando el cambio a la interfaz y gráficas.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

import ttkbootstrap as ttk

DEFAULT_THEME = "flatly"


class ThemeManager:
    """Administrador centralizado de temas de la interfaz gráfica."""

    def __init__(self, default_theme: str = DEFAULT_THEME) -> None:
        self._style = ttk.Style()
        names = self.available_themes()
        self._current_theme = (
            default_theme if default_theme in names else (names[0] if names else "flatly")
        )
        with contextlib.suppress(Exception):
            self._style.theme_use(self._current_theme)
        self._listeners: list[Callable[[str], None]] = []

    @property
    def current_theme(self) -> str:
        return self._current_theme

    @property
    def style(self) -> ttk.Style:
        return self._style

    def available_themes(self) -> list[str]:
        """Devuelve los nombres de temas disponibles en ttkbootstrap en tiempo de ejecución."""
        try:
            return sorted(self._style.theme_names())
        except Exception:
            return ["flatly", "darkly", "cosmo", "superhero", "litera", "solar"]

    def is_dark(self, theme_name: str | None = None) -> bool:
        """Determina si un tema es oscuro según el brillo de su fondo."""
        name = theme_name or self._current_theme
        with contextlib.suppress(Exception):
            colors = (
                self._style.theme_use(name).colors
                if name != self._current_theme
                else self._style.colors
            )
            bg = colors.bg
            if bg.startswith("#") and len(bg) == 7:
                r, g, b = int(bg[1:3], 16), int(bg[3:5], 16), int(bg[5:7], 16)
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                return luminance < 128
        return "dark" in name.lower() or name in {"darkly", "superhero", "cyborg", "solar", "vapor"}

    def add_theme_listener(self, callback: Callable[[str], None]) -> None:
        """Registra un callback que se invoca cuando el tema cambia exitosamente."""
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_theme_listener(self, callback: Callable[[str], None]) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    def apply_theme(self, theme_name: str) -> bool:
        """Aplica un nuevo tema a la aplicación y notifica a los suscriptores.

        Devuelve True si tuvo éxito, False en caso contrario.
        """
        available = self.available_themes()
        if theme_name not in available:
            return False

        try:
            self._style.theme_use(theme_name)
            self._current_theme = theme_name
            for listener in list(self._listeners):
                with contextlib.suppress(Exception):
                    listener(theme_name)
            return True
        except Exception:
            return False
