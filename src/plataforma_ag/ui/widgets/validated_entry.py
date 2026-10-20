"""Componente de entrada con validación visual reactiva (RF-10, TASK-047).

Proporciona retroalimentación visual inmediata (bootstyle danger y etiqueta inline)
y validación configurable al perder foco o con temporizador (debounce).
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from typing import Any

import ttkbootstrap as ttk


class ValidatedEntry(ttk.Frame):
    """Campo de texto con soporte para validación inline y bootstyle de error."""

    def __init__(
        self,
        master: tk.Widget,
        validator: Callable[[str], str | None] | None = None,
        on_validity_changed: Callable[[bool], None] | None = None,
        initial_value: str = "",
        width: int = 20,
        integer_only: bool = False,
        debounce_ms: int = 300,
        **entry_kwargs: Any,
    ) -> None:
        super().__init__(master)
        self.validator = validator
        self.on_validity_changed = on_validity_changed
        self.integer_only = integer_only
        self.debounce_ms = debounce_ms

        self._var = tk.StringVar(value=initial_value)
        self._error_message = tk.StringVar(value="")
        self._is_valid: bool = True
        self._debounce_job: str | None = None

        self._entry = ttk.Entry(
            self,
            textvariable=self._var,
            width=width,
            **entry_kwargs,
        )
        self._entry.pack(fill="x", expand=True)

        self._error_label = ttk.Label(
            self,
            textvariable=self._error_message,
            bootstyle="danger",
            font=("Segoe UI", 8),
            wraplength=200,
        )

        self._entry.bind("<FocusOut>", self._on_focus_out)
        self._var.trace_add("write", self._on_text_change)

        # Validación inicial silenciosa
        self.validate(silent=True)

    @property
    def value(self) -> str:
        return self._var.get()

    @value.setter
    def value(self, val: str) -> None:
        self._var.set(val)
        self.validate()

    @property
    def is_valid(self) -> bool:
        return self._is_valid

    @property
    def entry(self) -> ttk.Entry:
        return self._entry

    def set_state(self, state: str) -> None:
        """Configura el estado del widget interno (normal, disabled, readonly)."""
        self._entry.configure(state=state)

    def _on_focus_out(self, event: tk.Event[Any]) -> None:
        if self._debounce_job:
            self.after_cancel(self._debounce_job)
            self._debounce_job = None
        self.validate()

    def _on_text_change(self, *args: Any) -> None:
        if self._debounce_job:
            self.after_cancel(self._debounce_job)
        self._debounce_job = self.after(self.debounce_ms, self.validate)

    def validate(self, silent: bool = False) -> bool:
        """Ejecuta la función de validación y actualiza el estilo visual del campo."""
        self._debounce_job = None
        text = self._var.get()
        error: str | None = None

        if self.validator is not None:
            try:
                error = self.validator(text)
            except Exception as ex:
                error = str(ex)

        was_valid = self._is_valid
        self._is_valid = error is None

        if not silent:
            if error:
                self._entry.configure(bootstyle="danger")
                self._error_message.set(error)
                if not self._error_label.winfo_ismapped():
                    self._error_label.pack(anchor="w", pady=(2, 0))
            else:
                self._entry.configure(bootstyle="default")
                self._error_message.set("")
                if self._error_label.winfo_ismapped():
                    self._error_label.pack_forget()

        if was_valid != self._is_valid and self.on_validity_changed:
            self.on_validity_changed(self._is_valid)

        return self._is_valid
