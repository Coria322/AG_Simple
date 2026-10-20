"""Componente de barra de carga y progreso (RF-18, TASK-048, SUP-010).

Muestra el progreso de la simulación con máscara de generación y porcentaje,
estilos semánticos por estado y barra indeterminada de operaciones de E/S.
"""

from __future__ import annotations

import contextlib
import time
import tkinter as tk

import ttkbootstrap as ttk

from plataforma_ag.domain.enums import RunState
from plataforma_ag.domain.run import ProgressSnapshot


class ProgressView(ttk.Frame):
    """Visualizador de avance del algoritmo genético y estado de E/S."""

    def __init__(self, master: tk.Widget, padding: int = 6) -> None:
        super().__init__(master, padding=padding)

        # Barra principal con Floodgauge
        self._gauge_var = tk.DoubleVar(value=0.0)
        self._gauge = ttk.Floodgauge(
            self,
            variable=self._gauge_var,
            bootstyle="secondary",
            mode="determinate",
            mask="Listo",
            font=("Segoe UI", 9),
        )
        self._gauge.pack(fill="x", expand=True, side="left", padx=(0, 8))

        # Barra secundaria indeterminada para E/S (SUP-010)
        self._io_progress = ttk.Progressbar(
            self,
            bootstyle="info-striped",
            mode="indeterminate",
            length=120,
        )
        self._io_active: bool = False
        self._io_start_time: float = 0.0

        self.update_state(RunState.IDLE)

    def update_progress(self, snapshot: ProgressSnapshot) -> None:
        """Actualiza el valor porcentual y texto de la barra principal."""
        val = max(0.0, min(100.0, snapshot.percent))
        self._gauge_var.set(val)

        t = snapshot.current_generation
        gmax = snapshot.max_generations
        mask_text = f"Gen {t}/{gmax} · {val:.1f} %"
        self._gauge.configure(mask=mask_text)

    def update_state(self, state: RunState, convergence_note: str | None = None) -> None:
        """Actualiza el bootstyle y mensaje según el estado formal de ejecución."""
        if state is RunState.IDLE:
            self._gauge.configure(bootstyle="secondary", mask="Listo")
            self._gauge_var.set(0.0)
        elif state is RunState.RUNNING:
            self._gauge.configure(bootstyle="success-striped")
        elif state is RunState.PAUSED:
            curr_val = self._gauge_var.get()
            self._gauge.configure(bootstyle="warning-striped", mask=f"En pausa · {curr_val:.1f} %")
        elif state is RunState.COMPLETED:
            msg = f"Completada ({convergence_note})" if convergence_note else "Completada"
            self._gauge.configure(bootstyle="success", mask=msg)
            self._gauge_var.set(100.0)
        elif state is RunState.CANCELLED:
            self._gauge.configure(bootstyle="danger", mask="Abortada por el usuario")
        elif state is RunState.ERROR:
            self._gauge.configure(bootstyle="danger", mask="Error en la simulación")

    def show_io_activity(self) -> None:
        """Muestra la barra indeterminada de E/S."""
        self._io_active = True
        self._io_start_time = time.monotonic()
        if not self._io_progress.winfo_ismapped():
            self._io_progress.pack(side="right")
        self._io_progress.start(10)

    def hide_io_activity(self) -> None:
        """Oculta la barra indeterminada de E/S garantizando un mínimo de 300 ms visible."""
        if not self._io_active:
            return

        elapsed = time.monotonic() - self._io_start_time
        remaining = max(0.0, 0.3 - elapsed)

        if remaining > 0:
            self.after(int(remaining * 1000), self._stop_and_hide_io)
        else:
            self._stop_and_hide_io()

    def _stop_and_hide_io(self) -> None:
        self._io_active = False
        with contextlib.suppress(Exception):
            self._io_progress.stop()
            if self._io_progress.winfo_ismapped():
                self._io_progress.pack_forget()
