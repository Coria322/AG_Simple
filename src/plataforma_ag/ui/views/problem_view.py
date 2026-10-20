"""Panel de definición del problema de optimización (RF-01..RF-06, TASK-021..023, TASK-046).

Permite seleccionar funciones benchmark o expresiones libres, configurar
dimensiones N, sentido (Maximizar/Minimizar) y editar la tabla de dominios por variable.
"""

from __future__ import annotations

import contextlib
import tkinter as tk
from collections.abc import Callable
from tkinter import ttk as base_ttk
from typing import Any

import ttkbootstrap as ttk

from plataforma_ag.controllers.problem_controller import ProblemController
from plataforma_ag.core.problem.benchmarks import BENCHMARK_CATALOG
from plataforma_ag.domain.config import DEFAULT_BITS, ProblemDefinition
from plataforma_ag.domain.enums import BenchmarkFunction, ChromosomeType, ObjectiveSense
from plataforma_ag.ui.widgets.validated_entry import ValidatedEntry


class ProblemView(ttk.Frame):
    """Pestaña de configuración de función objetivo, dimensiones y límites."""

    def __init__(
        self,
        master: tk.Widget,
        controller: ProblemController,
        on_validation_changed: Callable[[bool], None] | None = None,
        padding: int = 12,
    ) -> None:
        super().__init__(master, padding=padding)
        self.controller = controller
        self.on_validation_changed = on_validation_changed

        self._current_chromosome: ChromosomeType = ChromosomeType.BINARY
        self._is_expression_valid: bool = True

        self._build_benchmark_section()
        self._build_dimensions_and_sense_section()
        self._build_bounds_table_section()

        # Suscribirse al controlador
        self.controller.add_listener(self._on_model_changed)
        self._refresh_from_model()

    def set_chromosome_type(self, chrom_type: ChromosomeType) -> None:
        """Adapta la visibilidad de la columna bits según la representación cromosómica."""
        self._current_chromosome = chrom_type
        self._update_bounds_table_display()

    def _build_benchmark_section(self) -> None:
        sec = ttk.LabelFrame(self, text="Función Objetivo (RF-01, RF-02)", padding=10)
        sec.pack(fill="x", pady=(0, 10))

        ttk.Label(sec, text="Función o Benchmark:").grid(row=0, column=0, sticky="w", pady=2)

        self._benchmark_labels = [spec.label for spec in BENCHMARK_CATALOG.values()] + [
            "Expresión personalizada"
        ]
        self._benchmark_combo = ttk.Combobox(
            sec,
            values=self._benchmark_labels,
            state="readonly",
            width=28,
        )
        self._benchmark_combo.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=2)
        self._benchmark_combo.bind("<<ComboboxSelected>>", self._on_benchmark_selected)

        # Fórmula visual
        self._lbl_formula = ttk.Label(
            sec,
            text="",
            font=("Segoe UI", 8, "italic"),
            bootstyle="secondary",
            wraplength=380,
        )
        self._lbl_formula.grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 6))

        # Campo de expresión libre
        ttk.Label(sec, text="Expresión f(x):").grid(row=2, column=0, sticky="w", pady=2)
        self._expr_entry = ValidatedEntry(
            sec,
            validator=self._validate_expression_text,
            on_validity_changed=self._on_expr_validity_changed,
            width=28,
        )
        self._expr_entry.grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=2)

        # Etiqueta obligatoria de radianes (RF-03)
        ttk.Label(
            sec,
            text="Ángulos: radianes (funciones trigonométricas esperan rad)",
            font=("Segoe UI", 8),
            bootstyle="info",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(2, 0))

        sec.columnconfigure(1, weight=1)

    def _build_dimensions_and_sense_section(self) -> None:
        sec = ttk.LabelFrame(self, text="Dimensiones y Sentido (RF-04, RF-05)", padding=10)
        sec.pack(fill="x", pady=(0, 10))

        # N dimensiones
        ttk.Label(sec, text="Dimensiones (N):").grid(row=0, column=0, sticky="w", pady=4)
        self._dim_var = tk.IntVar(value=self.controller.problem.dimensions)
        self._dim_spinbox = ttk.Spinbox(
            sec,
            from_=1,
            to=50,
            textvariable=self._dim_var,
            width=6,
            command=self._on_dimensions_spinbox,
        )
        self._dim_spinbox.grid(row=0, column=1, sticky="w", padx=(8, 16), pady=4)
        self._dim_spinbox.bind("<Return>", lambda e: self._on_dimensions_spinbox())

        # Sentido de optimización
        ttk.Label(sec, text="Sentido:").grid(row=0, column=2, sticky="w", pady=4)
        self._sense_var = tk.StringVar(value=self.controller.problem.objective.value)
        r_min = ttk.Radiobutton(
            sec,
            text="Minimizar",
            value=ObjectiveSense.MINIMIZE.value,
            variable=self._sense_var,
            command=self._on_sense_changed,
        )
        r_max = ttk.Radiobutton(
            sec,
            text="Maximizar",
            value=ObjectiveSense.MAXIMIZE.value,
            variable=self._sense_var,
            command=self._on_sense_changed,
        )
        r_min.grid(row=0, column=3, padx=4)
        r_max.grid(row=0, column=4, padx=4)

    def _build_bounds_table_section(self) -> None:
        sec = ttk.LabelFrame(self, text="Dominios y Límites por Variable (RF-05)", padding=10)
        sec.pack(fill="both", expand=True)

        table_frame = ttk.Frame(sec)
        table_frame.pack(fill="both", expand=True)

        self._bounds_tree = base_ttk.Treeview(
            table_frame,
            columns=("var", "min", "max", "bits"),
            show="headings",
            height=6,
        )
        self._bounds_tree.heading("var", text="Variable")
        self._bounds_tree.heading("min", text="Límite Min")
        self._bounds_tree.heading("max", text="Límite Max")
        self._bounds_tree.heading("bits", text="Bits (m)")

        self._bounds_tree.column("var", width=70, anchor="center")
        self._bounds_tree.column("min", width=90, anchor="e")
        self._bounds_tree.column("max", width=90, anchor="e")
        self._bounds_tree.column("bits", width=70, anchor="center")

        v_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self._bounds_tree.yview)
        self._bounds_tree.configure(yscrollcommand=v_scroll.set)

        self._bounds_tree.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        # Editor rápido bajo la tabla
        edit_box = ttk.Frame(sec)
        edit_box.pack(fill="x", pady=(6, 0))

        ttk.Label(edit_box, text="Editar selección: Min:").pack(side="left", padx=(0, 4))
        self._edit_min_var = tk.StringVar(value="-5.12")
        self._edit_min = ttk.Entry(edit_box, textvariable=self._edit_min_var, width=8)
        self._edit_min.pack(side="left", padx=(0, 6))

        ttk.Label(edit_box, text="Max:").pack(side="left", padx=(0, 4))
        self._edit_max_var = tk.StringVar(value="5.12")
        self._edit_max = ttk.Entry(edit_box, textvariable=self._edit_max_var, width=8)
        self._edit_max.pack(side="left", padx=(0, 6))

        self._lbl_edit_bits = ttk.Label(edit_box, text="Bits:")
        self._lbl_edit_bits.pack(side="left", padx=(0, 4))
        self._edit_bits_var = tk.StringVar(value="16")
        self._edit_bits = ttk.Entry(edit_box, textvariable=self._edit_bits_var, width=6)
        self._edit_bits.pack(side="left", padx=(0, 6))

        btn_apply = ttk.Button(
            edit_box,
            text="Aplicar",
            bootstyle="primary-outline",
            command=self._apply_bound_edit,
        )
        btn_apply.pack(side="left", padx=(4, 0))

        self._bounds_tree.bind("<<TreeviewSelect>>", self._on_tree_select)

    def _on_benchmark_selected(self, event: Any = None) -> None:
        idx = self._benchmark_combo.current()
        catalogs = list(BENCHMARK_CATALOG.values())
        if idx < len(catalogs):
            func = catalogs[idx].name
            self.controller.set_benchmark(func)
            self._expr_entry.set_state("disabled")
            self._lbl_formula.configure(text=f"Fórmula: {catalogs[idx].formula_latex}")
        else:
            self.controller.set_benchmark(BenchmarkFunction.CUSTOM_EXPRESSION)
            self._expr_entry.set_state("normal")
            self._lbl_formula.configure(text="Expresión matemática libre ingresada por el usuario.")

    def _on_dimensions_spinbox(self) -> None:
        with contextlib.suppress(Exception):
            val = int(self._dim_var.get())
            if val >= 1:
                self.controller.set_dimensions(val)

    def _on_sense_changed(self) -> None:
        val = self._sense_var.get()
        self.controller.set_objective_sense(ObjectiveSense(val))

    def _validate_expression_text(self, text: str) -> str | None:
        if self.controller.problem.benchmark is BenchmarkFunction.CUSTOM_EXPRESSION:
            return self.controller.validate_expression(text)
        return None

    def _on_expr_validity_changed(self, is_valid: bool) -> None:
        self._is_expression_valid = is_valid
        if is_valid and self.controller.problem.benchmark is BenchmarkFunction.CUSTOM_EXPRESSION:
            self.controller.set_expression(self._expr_entry.value)
        if self.on_validation_changed:
            self.on_validation_changed(is_valid)

    def _on_tree_select(self, event: Any) -> None:
        selected = self._bounds_tree.selection()
        if not selected:
            return
        item_vals = self._bounds_tree.item(selected[0], "values")
        if item_vals:
            self._edit_min_var.set(item_vals[1])
            self._edit_max_var.set(item_vals[2])
            if len(item_vals) > 3:
                self._edit_bits_var.set(item_vals[3])

    def _apply_bound_edit(self) -> None:
        selected = self._bounds_tree.selection()
        if not selected:
            return
        idx = self._bounds_tree.index(selected[0])
        with contextlib.suppress(Exception):
            min_val = float(self._edit_min_var.get().replace(",", "."))
            max_val = float(self._edit_max_var.get().replace(",", "."))
            bits_val = int(self._edit_bits_var.get()) if self._edit_bits_var.get() else DEFAULT_BITS

            if min_val >= max_val:
                return

            self.controller.update_variable_bound(idx, min_val, max_val, bits_val)

    def _update_bounds_table_display(self) -> None:
        is_binary_or_gray = self._current_chromosome in (ChromosomeType.BINARY, ChromosomeType.GRAY)
        if is_binary_or_gray:
            self._bounds_tree["columns"] = ("var", "min", "max", "bits")
            self._lbl_edit_bits.pack(side="left", padx=(0, 4))
            self._edit_bits.pack(side="left", padx=(0, 6))
        else:
            self._bounds_tree["columns"] = ("var", "min", "max")
            self._lbl_edit_bits.pack_forget()
            self._edit_bits.pack_forget()

        # Re-poblar filas
        for row in self._bounds_tree.get_children():
            self._bounds_tree.delete(row)

        for b in self.controller.problem.bounds:
            vals: tuple[str, ...]
            if is_binary_or_gray:
                vals = (b.variable, f"{b.min:.4g}", f"{b.max:.4g}", str(b.bits))
            else:
                vals = (b.variable, f"{b.min:.4g}", f"{b.max:.4g}")
            self._bounds_tree.insert("", "end", values=vals)

    def _on_model_changed(self, problem: ProblemDefinition) -> None:
        self._refresh_from_model()

    def _refresh_from_model(self) -> None:
        prob = self.controller.problem
        # Benchmark combo
        if prob.benchmark is BenchmarkFunction.CUSTOM_EXPRESSION:
            self._benchmark_combo.set("Expresión personalizada")
            self._expr_entry.set_state("normal")
            self._lbl_formula.configure(text="Expresión matemática libre ingresada por el usuario.")
        else:
            spec = BENCHMARK_CATALOG.get(prob.benchmark)
            if spec:
                self._benchmark_combo.set(spec.label)
                self._lbl_formula.configure(text=f"Fórmula: {spec.formula_latex}")
            self._expr_entry.set_state("disabled")

        if prob.expression:
            self._expr_entry.value = prob.expression

        self._dim_var.set(prob.dimensions)
        self._sense_var.set(prob.objective.value)
        self._update_bounds_table_display()
