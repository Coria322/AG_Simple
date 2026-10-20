"""Tabla de población paginada con selector de generación (RF-20, TASK-050).

Permite inspeccionar los individuos de cualquier generación disponible,
con genotipos, fenotipos (x1..xN), fitness, marca de élite y ordenamiento.
"""

from __future__ import annotations

import contextlib
import math
import tkinter as tk
from tkinter import ttk as base_ttk
from typing import Any

import ttkbootstrap as ttk

from plataforma_ag.controllers.telemetry_controller import TelemetryController


class PopulationTable(ttk.Frame):
    """Vista de tabla paginada para inspección granular de generaciones de individuos."""

    def __init__(
        self,
        master: tk.Widget,
        telemetry_controller: TelemetryController,
        padding: int = 6,
    ) -> None:
        super().__init__(master, padding=padding)
        self.telemetry = telemetry_controller

        self._current_generation: int = 0
        self._max_available_generation: int = 0
        self._page: int = 1
        self._page_size: int = 25
        self._sort_by: str = "id"
        self._ascending: bool = True
        self._total_individuals: int = 0

        # Barra superior de controles: selector de generación y paginación
        top_bar = ttk.Frame(self)
        top_bar.pack(fill="x", pady=(0, 6))

        ttk.Label(top_bar, text="Generación:", font=("Segoe UI", 9, "bold")).pack(
            side="left", padx=(0, 4)
        )
        self._gen_var = tk.IntVar(value=0)
        self._gen_spinbox = ttk.Spinbox(
            top_bar,
            from_=0,
            to=0,
            textvariable=self._gen_var,
            width=6,
            command=self._on_generation_changed,
        )
        self._gen_spinbox.pack(side="left", padx=(0, 16))
        self._gen_spinbox.bind("<Return>", lambda e: self._on_generation_changed())

        # Botones de página
        self._btn_prev = ttk.Button(
            top_bar,
            text="◀ Ant",
            bootstyle="secondary-outline",
            width=6,
            command=self._prev_page,
        )
        self._btn_prev.pack(side="left", padx=(0, 4))

        self._lbl_page = ttk.Label(top_bar, text="Pág 1 / 1")
        self._lbl_page.pack(side="left", padx=4)

        self._btn_next = ttk.Button(
            top_bar,
            text="Sig ▶",
            bootstyle="secondary-outline",
            width=6,
            command=self._next_page,
        )
        self._btn_next.pack(side="left", padx=(4, 16))

        # Ordenamiento
        ttk.Label(top_bar, text="Ordenar por:").pack(side="left", padx=(0, 4))
        self._sort_combo = ttk.Combobox(
            top_bar,
            values=[
                "ID (Ascendente)",
                "ID (Descendente)",
                "Fitness (Ascendente)",
                "Fitness (Descendente)",
            ],
            state="readonly",
            width=20,
        )
        self._sort_combo.current(0)
        self._sort_combo.pack(side="left")
        self._sort_combo.bind("<<ComboboxSelected>>", self._on_sort_changed)

        # Contenedor central con Treeview
        tree_container = ttk.Frame(self)
        tree_container.pack(fill="both", expand=True)

        self._tree = base_ttk.Treeview(
            tree_container,
            columns=("id", "genotype", "fitness", "elite"),
            show="headings",
            selectmode="browse",
        )
        self._tree.heading("id", text="ID")
        self._tree.heading("genotype", text="Genotipo")
        self._tree.heading("fitness", text="Fitness f(x)")
        self._tree.heading("elite", text="Élite")

        self._tree.column("id", width=60, anchor="center")
        self._tree.column("genotype", width=160, anchor="w")
        self._tree.column("fitness", width=120, anchor="e")
        self._tree.column("elite", width=60, anchor="center")

        v_scroll = ttk.Scrollbar(tree_container, orient="vertical", command=self._tree.yview)
        h_scroll = ttk.Scrollbar(tree_container, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        self._tree.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")

        tree_container.grid_rowconfigure(0, weight=1)
        tree_container.grid_columnconfigure(0, weight=1)

        # Etiqueta de estado vacío
        self._empty_label = ttk.Label(
            self,
            text="Sin corridas ejecutadas. Inicia una simulación o importa un CSV.",
            bootstyle="secondary",
            font=("Segoe UI", 9, "italic"),
        )
        self._empty_label.pack(pady=12)

        self.refresh()

    def update_max_generation(self, max_gen: int) -> None:
        """Actualiza el rango del selector al llegar nuevas generaciones sin saltar selección."""
        self._max_available_generation = max(0, max_gen)
        self._gen_spinbox.configure(to=self._max_available_generation)
        if self._empty_label.winfo_ismapped():
            self._empty_label.pack_forget()

    def _on_generation_changed(self) -> None:
        with contextlib.suppress(Exception):
            val = self._gen_var.get()
            val = max(0, min(self._max_available_generation, val))
            self._current_generation = val
            self._page = 1
            self.refresh()

    def _on_sort_changed(self, event: Any) -> None:
        idx = self._sort_combo.current()
        if idx == 0:
            self._sort_by = "id"
            self._ascending = True
        elif idx == 1:
            self._sort_by = "id"
            self._ascending = False
        elif idx == 2:
            self._sort_by = "fitness"
            self._ascending = True
        elif idx == 3:
            self._sort_by = "fitness"
            self._ascending = False
        self.refresh()

    def _prev_page(self) -> None:
        if self._page > 1:
            self._page -= 1
            self.refresh()

    def _next_page(self) -> None:
        total_pages = max(1, math.ceil(self._total_individuals / self._page_size))
        if self._page < total_pages:
            self._page += 1
            self.refresh()

    def refresh(self) -> None:
        """Recarga la página de individuos para la generación y orden actual."""
        items, total = self.telemetry.get_population_page(
            generation=self._current_generation,
            page=self._page,
            page_size=self._page_size,
            sort_by=self._sort_by,
            ascending=self._ascending,
        )
        self._total_individuals = total

        # Actualizar columnas de fenotipos si hay datos
        if items:
            num_vars = len(items[0].phenotype)
            col_ids = (
                ["id", "genotype"] + [f"x{i + 1}" for i in range(num_vars)] + ["fitness", "elite"]
            )
            self._tree["columns"] = col_ids
            self._tree.heading("id", text="ID")
            self._tree.heading("genotype", text="Genotipo")
            self._tree.column("id", width=50, anchor="center")
            self._tree.column("genotype", width=140, anchor="w")

            for i in range(num_vars):
                cid = f"x{i + 1}"
                self._tree.heading(cid, text=cid)
                self._tree.column(cid, width=80, anchor="e")

            self._tree.heading("fitness", text="Fitness f(x)")
            self._tree.heading("elite", text="Élite")
            self._tree.column("fitness", width=100, anchor="e")
            self._tree.column("elite", width=50, anchor="center")

        # Limpiar filas existentes
        for row in self._tree.get_children():
            self._tree.delete(row)

        if not items:
            if not self._empty_label.winfo_ismapped() and self._max_available_generation == 0:
                self._empty_label.pack(pady=12)
            self._lbl_page.configure(text="Pág 1 / 1")
            self._btn_prev.configure(state="disabled")
            self._btn_next.configure(state="disabled")
            return

        if self._empty_label.winfo_ismapped():
            self._empty_label.pack_forget()

        # Insertar registros
        for ind in items:
            geno_str = "".join(str(g) for g in ind.genotype) if ind.genotype else "—"
            if len(geno_str) > 24:
                geno_str = geno_str[:22] + "…"
            pheno_vals = [f"{p:.5g}" for p in ind.phenotype]
            fitness_str = f"{ind.fitness:.6g}" if math.isfinite(ind.fitness) else str(ind.fitness)
            elite_str = "✔" if ind.is_elite else ""

            row_vals = [ind.individual_id, geno_str, *pheno_vals, fitness_str, elite_str]
            self._tree.insert("", "end", values=row_vals)

        # Controles de paginación
        total_pages = max(1, math.ceil(total / self._page_size))
        self._lbl_page.configure(text=f"Pág {self._page} / {total_pages} (Total: {total})")
        self._btn_prev.configure(state="normal" if self._page > 1 else "disabled")
        self._btn_next.configure(state="normal" if self._page < total_pages else "disabled")
