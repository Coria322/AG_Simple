"""Panel de representación cromosómica y operadores genéticos (RF-11..RF-16, TASK-024, TASK-025).

Permite seleccionar la representación (Binaria, Gray, Real, Permutaciones) y
habilita dinámicamente los operadores compatibles de selección, cruce, mutación
y reemplazo, junto con sus parámetros específicos (k, s, E, etc.).
"""

from __future__ import annotations

import contextlib
import tkinter as tk
from collections.abc import Callable
from typing import Any

import ttkbootstrap as ttk
from ttkbootstrap import ToastNotification

from plataforma_ag.controllers.params_controller import ParamsController
from plataforma_ag.domain.compatibility import COMPATIBILITY_MATRIX
from plataforma_ag.domain.enums import (
    ChromosomeType,
    CrossoverMethod,
    MutationMethod,
    ReplacementMethod,
    SelectionMethod,
)


class OperatorsView(ttk.Frame):
    """Pestaña de operadores genéticos y compatibilidad."""

    def __init__(
        self,
        master: tk.Widget,
        controller: ParamsController,
        on_representation_changed: Callable[[ChromosomeType], None] | None = None,
        padding: int = 12,
    ) -> None:
        super().__init__(master, padding=padding)
        self.controller = controller
        self.on_representation_changed = on_representation_changed

        self._build_representation_section()
        self._build_operators_section()
        self._build_operator_params_section()

        self.controller.add_listener(self._on_model_changed)
        self.controller.add_operator_notification_listener(self._show_operator_toast)
        self._refresh_from_model()

    def _show_operator_toast(self, kind: str, new_operator: str) -> None:
        """Muestra una notificación flotante cuando un operador cambia por compatibilidad."""
        with contextlib.suppress(Exception):
            toast = ToastNotification(
                title="Operador ajustado",
                message=(
                    f"El método de {kind} cambió automáticamente a "
                    f"'{new_operator}' para mantener compatibilidad."
                ),
                duration=3000,
                bootstyle="info",
            )
            toast.show_toast()

    def _build_representation_section(self) -> None:
        sec = ttk.LabelFrame(self, text="Representación Cromosómica (RF-11)", padding=10)
        sec.pack(fill="x", pady=(0, 10))

        ttk.Label(sec, text="Tipo de Cromosoma:").grid(row=0, column=0, sticky="w", pady=4)
        self._chrom_values = [
            ChromosomeType.BINARY.value,
            ChromosomeType.GRAY.value,
            ChromosomeType.REAL.value,
            ChromosomeType.PERMUTATION.value,
        ]
        self._chrom_combo = ttk.Combobox(
            sec,
            values=self._chrom_values,
            state="readonly",
            width=22,
        )
        self._chrom_combo.grid(row=0, column=1, sticky="w", padx=(8, 0), pady=4)
        self._chrom_combo.bind("<<ComboboxSelected>>", self._on_chrom_selected)

    def _build_operators_section(self) -> None:
        sec = ttk.LabelFrame(self, text="Operadores Genéticos (RF-12..RF-16)", padding=10)
        sec.pack(fill="x", pady=(0, 10))

        # Selección
        ttk.Label(sec, text="Selección:").grid(row=0, column=0, sticky="w", pady=4)
        self._sel_combo = ttk.Combobox(sec, state="readonly", width=32)
        self._sel_combo.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=4)
        self._sel_combo.bind("<<ComboboxSelected>>", self._on_operator_selected)

        # Cruce
        ttk.Label(sec, text="Cruce:").grid(row=1, column=0, sticky="w", pady=4)
        self._cx_combo = ttk.Combobox(sec, state="readonly", width=32)
        self._cx_combo.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=4)
        self._cx_combo.bind("<<ComboboxSelected>>", self._on_operator_selected)

        # Mutación
        ttk.Label(sec, text="Mutación:").grid(row=2, column=0, sticky="w", pady=4)
        self._mut_combo = ttk.Combobox(sec, state="readonly", width=32)
        self._mut_combo.grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=4)
        self._mut_combo.bind("<<ComboboxSelected>>", self._on_operator_selected)

        # Reemplazo
        ttk.Label(sec, text="Esquema de Reemplazo:").grid(row=3, column=0, sticky="w", pady=4)
        self._rep_combo = ttk.Combobox(sec, state="readonly", width=32)
        self._rep_combo.grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=4)
        self._rep_combo.bind("<<ComboboxSelected>>", self._on_operator_selected)

        sec.columnconfigure(1, weight=1)

    def _build_operator_params_section(self) -> None:
        sec = ttk.LabelFrame(self, text="Parámetros Específicos de Operadores", padding=10)
        sec.pack(fill="x", pady=(0, 10))

        # Parámetro k de Torneo
        self._lbl_k = ttk.Label(sec, text="Tamaño de torneo (k):")
        self._lbl_k.grid(row=0, column=0, sticky="w", pady=4)
        self._k_var = tk.IntVar(value=3)
        self._k_spin = ttk.Spinbox(
            sec,
            from_=2,
            to=20,
            textvariable=self._k_var,
            width=6,
            command=self._on_params_entry_changed,
        )
        self._k_spin.grid(row=0, column=1, sticky="w", padx=(8, 16), pady=4)

        # Elitismo count
        self._lbl_elitism = ttk.Label(sec, text="Individuos élite (E):")
        self._lbl_elitism.grid(row=0, column=2, sticky="w", pady=4)
        self._elitism_var = tk.IntVar(value=2)
        self._elitism_spin = ttk.Spinbox(
            sec,
            from_=0,
            to=10,
            textvariable=self._elitism_var,
            width=6,
            command=self._on_params_entry_changed,
        )
        self._elitism_spin.grid(row=0, column=3, sticky="w", padx=(8, 0), pady=4)

        # Crowding Factor
        self._lbl_cf = ttk.Label(sec, text="Factor de crowding (CF):")
        self._lbl_cf.grid(row=1, column=0, sticky="w", pady=4)
        self._cf_var = tk.IntVar(value=3)
        self._cf_spin = ttk.Spinbox(
            sec,
            from_=2,
            to=10,
            textvariable=self._cf_var,
            width=6,
            command=self._on_params_entry_changed,
        )
        self._cf_spin.grid(row=1, column=1, sticky="w", padx=(8, 16), pady=4)

    def _on_chrom_selected(self, event: Any = None) -> None:
        val = self._chrom_combo.get()
        chrom = ChromosomeType(val)
        self.controller.set_chromosome_type(chrom)
        if self.on_representation_changed:
            self.on_representation_changed(chrom)

    def _on_operator_selected(self, event: Any = None) -> None:
        sel_val = self._sel_combo.get()
        cx_val = self._cx_combo.get()
        mut_val = self._mut_combo.get()
        rep_val = self._rep_combo.get()

        k_val = int(self._k_var.get())
        e_val = int(self._elitism_var.get())
        cf_val = int(self._cf_var.get())

        if sel_val:
            self.controller.set_selection_method(SelectionMethod(sel_val), params={"k": k_val})
        if cx_val:
            self.controller.set_crossover_method(CrossoverMethod(cx_val))
        if mut_val:
            self.controller.set_mutation_method(MutationMethod(mut_val))
        if rep_val:
            self.controller.set_replacement_method(
                ReplacementMethod(rep_val),
                elitism_count=e_val,
                crowding_factor=cf_val if rep_val == ReplacementMethod.CROWDING.value else None,
            )

    def _on_params_entry_changed(self) -> None:
        self._on_operator_selected()

    def _on_model_changed(self) -> None:
        self._refresh_from_model()

    def _refresh_from_model(self) -> None:
        chrom = self.controller.genetic.chromosome_type
        self._chrom_combo.set(chrom.value)

        rule = COMPATIBILITY_MATRIX[chrom]

        # Poblar y habilitar sólo métodos compatibles
        sel_names = sorted(m.value for m in rule.allowed_selection)
        cx_names = sorted(m.value for m in rule.allowed_crossover)
        mut_names = sorted(m.value for m in rule.allowed_mutation)
        rep_names = sorted(m.value for m in rule.allowed_replacement)

        self._sel_combo.configure(values=sel_names)
        self._cx_combo.configure(values=cx_names)
        self._mut_combo.configure(values=mut_names)
        self._rep_combo.configure(values=rep_names)

        curr_ops = self.controller.operators
        self._sel_combo.set(curr_ops.selection_method.value)
        self._cx_combo.set(curr_ops.crossover_method.value)
        self._mut_combo.set(curr_ops.mutation_method.value)
        self._rep_combo.set(curr_ops.replacement_method.value)

        # Ajustar visibilidad de parámetros dinámicos
        is_tournament = curr_ops.selection_method is SelectionMethod.TOURNAMENT
        if is_tournament:
            self._lbl_k.grid()
            self._k_spin.grid()
            self._k_var.set(curr_ops.selection_params.get("k", 3))
        else:
            self._lbl_k.grid_remove()
            self._k_spin.grid_remove()

        is_crowding = curr_ops.replacement_method is ReplacementMethod.CROWDING
        if is_crowding:
            self._lbl_cf.grid()
            self._cf_spin.grid()
            self._cf_var.set(curr_ops.crowding_factor or 3)
        else:
            self._lbl_cf.grid_remove()
            self._cf_spin.grid_remove()

        self._elitism_var.set(curr_ops.elitism_count or 0)
