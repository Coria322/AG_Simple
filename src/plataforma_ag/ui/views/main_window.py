"""Ventana principal (shell provisional del Sprint 1).

TASK-020 la reemplaza por el layout completo de ``UI_UX_SPEC.md``.
"""

from __future__ import annotations

import ttkbootstrap as ttk

from plataforma_ag import __version__

DEFAULT_THEME = "flatly"  # SUP-009: tema por defecto; la lista se lee de Style().theme_names()


class MainWindow:
    """Shell mínimo con ttkbootstrap."""

    def __init__(self, themename: str = DEFAULT_THEME) -> None:
        self.root = ttk.Window(
            title=f"Plataforma AG {__version__}",
            themename=themename,
            minsize=(960, 600),
        )
        frame = ttk.Frame(self.root, padding=24)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text="Plataforma de Simulación y Optimización con Algoritmos Genéticos",
            font=("Segoe UI", 16, "bold"),
            bootstyle="primary",
        ).pack(anchor="w")
        ttk.Label(
            frame,
            text="Sprint 1 — fundaciones del núcleo. La interfaz completa llega en TASK-020.",
            bootstyle="secondary",
        ).pack(anchor="w", pady=(8, 0))

    def run(self) -> None:
        self.root.mainloop()
