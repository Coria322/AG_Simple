"""Barra de control de ejecución del AG (RF-17, TASK-027).

Implementa los botones Iniciar, Pausar, Reanudar y Abortar según la tabla
de estados finitos de UI_UX_SPEC.md y diálogo modal de confirmación.
"""

from __future__ import annotations

import time
import tkinter as tk
from collections.abc import Callable

import ttkbootstrap as ttk
from ttkbootstrap.dialogs import Messagebox

from plataforma_ag.domain.enums import RunState


class ControlBar(ttk.Frame):
    """Barra de botones de control para iniciar, pausar, reanudar y abortar."""

    def __init__(
        self,
        master: tk.Widget,
        on_start: Callable[[], None] | None = None,
        on_pause: Callable[[], None] | None = None,
        on_resume: Callable[[], None] | None = None,
        on_abort: Callable[[], None] | None = None,
        padding: int = 8,
    ) -> None:
        super().__init__(master, padding=padding)
        self.on_start = on_start
        self.on_pause = on_pause
        self.on_resume = on_resume
        self.on_abort = on_abort

        self._current_state = RunState.IDLE
        self._is_form_valid: bool = True
        self._last_start_click_time: float = 0.0

        self._btn_start = ttk.Button(
            self,
            text="▶ Iniciar",
            bootstyle="success",
            command=self._handle_start_click,
            width=12,
        )
        self._btn_start.pack(side="left", padx=(0, 6))

        self._btn_pause = ttk.Button(
            self,
            text="⏸ Pausar",
            bootstyle="warning",
            command=self._handle_pause_click,
            width=12,
        )
        self._btn_pause.pack(side="left", padx=(0, 6))

        self._btn_resume = ttk.Button(
            self,
            text="⏯ Reanudar",
            bootstyle="info",
            command=self._handle_resume_click,
            width=12,
        )
        self._btn_resume.pack(side="left", padx=(0, 6))

        self._btn_abort = ttk.Button(
            self,
            text="⏹ Abortar",
            bootstyle="danger",
            command=self._handle_abort_click,
            width=12,
        )
        self._btn_abort.pack(side="left", padx=(0, 6))

        self.update_state(RunState.IDLE)

    def set_form_valid(self, is_valid: bool) -> None:
        """Indica si los formularios de configuración tienen errores de validación."""
        self._is_form_valid = is_valid
        if self._current_state in (
            RunState.IDLE,
            RunState.COMPLETED,
            RunState.CANCELLED,
            RunState.ERROR,
        ):
            self._btn_start.configure(state="normal" if is_valid else "disabled")

    def update_state(self, state: RunState) -> None:
        """Actualiza la habilitación de botones conforme a la matriz de UI_UX_SPEC."""
        self._current_state = state

        if state is RunState.RUNNING:
            self._btn_start.configure(state="disabled")
            self._btn_pause.configure(state="normal")
            self._btn_resume.configure(state="disabled")
            self._btn_abort.configure(state="normal")
        elif state is RunState.PAUSED:
            self._btn_start.configure(state="disabled")
            self._btn_pause.configure(state="disabled")
            self._btn_resume.configure(state="normal")
            self._btn_abort.configure(state="normal")
        else:  # IDLE, COMPLETED, CANCELLED, ERROR
            self._btn_start.configure(state="normal" if self._is_form_valid else "disabled")
            self._btn_pause.configure(state="disabled")
            self._btn_resume.configure(state="disabled")
            self._btn_abort.configure(state="disabled")

    def _handle_start_click(self) -> None:
        # Ignorar doble clic rápido accidental (debounce de 500 ms)
        now = time.monotonic()
        if now - self._last_start_click_time < 0.5:
            return
        self._last_start_click_time = now

        if self.on_start and self._is_form_valid:
            self.on_start()

    def _handle_pause_click(self) -> None:
        if self.on_pause:
            self.on_pause()

    def _handle_resume_click(self) -> None:
        if self.on_resume:
            self.on_resume()

    def _handle_abort_click(self) -> None:
        # Confirmación obligatoria según RF-17
        ans = Messagebox.yesno(
            message="¿Abortar la simulación? Se conservarán los datos acumulados.",
            title="Confirmar aborto",
            parent=self.winfo_toplevel(),
        )
        if ans == "Yes" and self.on_abort:
            self.on_abort()
