"""Consola visual de bitácora con autoscroll inteligente (RF-19, TASK-049).

Muestra eventos estructurados con timestamps en milisegundos y colores semánticos
según el nivel (INFO, WARN, ERROR), pausando el autoscroll si el usuario sube.
"""

from __future__ import annotations

import tkinter as tk
from datetime import UTC, datetime
from typing import Any

import ttkbootstrap as ttk
from ttkbootstrap import ScrolledText

from plataforma_ag.domain.events import LogRecorded
from plataforma_ag.domain.run import LogEvent


class LogConsole(ttk.Frame):
    """Consola de bitácora enriquecida con etiquetas de color y control de scroll."""

    def __init__(self, master: tk.Widget, max_lines: int = 1000, padding: int = 6) -> None:
        super().__init__(master, padding=padding)
        self.max_lines = max_lines
        self._user_scrolled_up = False

        # Barra superior con botones de ayuda
        header = ttk.Frame(self)
        header.pack(fill="x", pady=(0, 4))
        ttk.Label(header, text="Bitácora de Eventos", font=("Segoe UI", 9, "bold")).pack(
            side="left"
        )

        btn_clear = ttk.Button(
            header,
            text="Limpiar",
            bootstyle="secondary-outline",
            command=self.clear,
            width=8,
        )
        btn_clear.pack(side="right")

        # ScrolledText
        self._text = ScrolledText(
            self,
            autohide=True,
            height=8,
            font=("Consolas", 9),
            wrap="none",
        )
        self._text.pack(fill="both", expand=True)

        # Configurar tags de color
        self._configure_tags()

        # Monitorear scroll del usuario para pausar autoscroll
        self._text.text.bind("<Button-4>", self._on_scroll_up)
        self._text.text.bind("<Button-5>", self._on_scroll_down)
        self._text.text.bind("<MouseWheel>", self._on_mousewheel)

    def _configure_tags(self) -> None:
        text_widget = self._text.text
        text_widget.tag_configure("INFO", foreground="#28a745")
        text_widget.tag_configure("WARN", foreground="#ffc107")
        text_widget.tag_configure("ERROR", foreground="#dc3545", font=("Consolas", 9, "bold"))
        text_widget.tag_configure("TIMESTAMP", foreground="#6c757d")

    def _on_mousewheel(self, event: tk.Event[Any]) -> None:
        if event.delta > 0:
            self._user_scrolled_up = True
        else:
            # Comprobar si volvió al fondo
            yview = self._text.text.yview()
            if yview[1] >= 0.99:
                self._user_scrolled_up = False

    def _on_scroll_up(self, event: tk.Event[Any]) -> None:
        self._user_scrolled_up = True

    def _on_scroll_down(self, event: tk.Event[Any]) -> None:
        yview = self._text.text.yview()
        if yview[1] >= 0.99:
            self._user_scrolled_up = False

    def add_log_event(self, event: LogRecorded | LogEvent) -> None:
        """Agrega una línea formateada a la bitácora con su marca de tiempo y nivel."""
        text_widget = self._text.text
        dt = getattr(event, "occurred_at", getattr(event, "timestamp", datetime.now(UTC)))
        ts_str = dt.strftime("[%H:%M:%S.") + f"{dt.microsecond // 1000:03d}] "

        text_widget.configure(state="normal")

        # Control de límite de líneas (ring buffer visual)
        num_lines = int(text_widget.index("end-1c").split(".")[0])
        if num_lines > self.max_lines:
            text_widget.delete("1.0", f"{num_lines - self.max_lines}.0")

        level_name = event.level.value if hasattr(event.level, "value") else str(event.level)
        text_widget.insert("end", ts_str, ("TIMESTAMP",))
        text_widget.insert("end", f"[{level_name}] ", (level_name,))
        text_widget.insert("end", f"{event.message}\n")

        text_widget.configure(state="disabled")

        if not self._user_scrolled_up:
            text_widget.see("end")

    def clear(self) -> None:
        """Limpia el contenido de la consola."""
        text_widget = self._text.text
        text_widget.configure(state="normal")
        text_widget.delete("1.0", "end")
        text_widget.configure(state="disabled")
        self._user_scrolled_up = False
