"""Panel de parámetros genéticos y criterios de parada (RF-07..RF-09, TASK-026).

Permite configurar el tamaño de población Np, probabilidades Pc y Pm,
semilla aleatoria, y criterios de parada (Gmax y estancamiento).
"""

from __future__ import annotations

import contextlib
import tkinter as tk
from collections.abc import Callable

import ttkbootstrap as ttk

from plataforma_ag.controllers.params_controller import ParamsController
from plataforma_ag.domain.validation import validate_number_format
from plataforma_ag.ui.widgets.validated_entry import ValidatedEntry


class ParamsView(ttk.Frame):
    """Pestaña de población, probabilidades y criterios de parada."""

    def __init__(
        self,
        master: tk.Widget,
        controller: ParamsController,
        on_validation_changed: Callable[[bool], None] | None = None,
        padding: int = 12,
    ) -> None:
        super().__init__(master, padding=padding)
        self.controller = controller
        self.on_validation_changed = on_validation_changed

        self._build_genetic_params_section()
        self._build_stopping_criteria_section()

        self.controller.add_listener(self._on_model_changed)
        self._refresh_from_model()

    def _build_genetic_params_section(self) -> None:
        sec = ttk.LabelFrame(self, text="Parámetros Poblacionales (RF-07, RF-08)", padding=10)
        sec.pack(fill="x", pady=(0, 10))

        # Tamaño de población (Np)
        ttk.Label(sec, text="Tamaño de población (Np):").grid(row=0, column=0, sticky="w", pady=4)
        self._np_var = tk.IntVar(value=self.controller.genetic.population_size)
        self._np_spin = ttk.Spinbox(
            sec,
            from_=2,
            to=10000,
            textvariable=self._np_var,
            width=10,
            command=self._on_params_changed,
        )
        self._np_spin.grid(row=0, column=1, sticky="w", padx=(8, 0), pady=4)
        self._np_spin.bind("<Return>", lambda e: self._on_params_changed())

        # Probabilidad de cruce (Pc)
        ttk.Label(sec, text="Probabilidad de cruce (Pc):").grid(row=1, column=0, sticky="w", pady=4)
        self._pc_entry = ValidatedEntry(
            sec,
            validator=self._validate_prob,
            on_validity_changed=self._on_validity_updated,
            initial_value=str(self.controller.genetic.p_crossover),
            width=10,
        )
        self._pc_entry.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=4)

        # Probabilidad de mutación (Pm)
        ttk.Label(sec, text="Probabilidad de mutación (Pm):").grid(
            row=2, column=0, sticky="w", pady=4
        )
        self._pm_entry = ValidatedEntry(
            sec,
            validator=self._validate_prob,
            on_validity_changed=self._on_validity_updated,
            initial_value=str(self.controller.genetic.p_mutation),
            width=10,
        )
        self._pm_entry.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=4)

        # Semilla aleatoria opcional
        ttk.Label(sec, text="Semilla aleatoria (opcional):").grid(
            row=3, column=0, sticky="w", pady=4
        )
        self._seed_entry = ValidatedEntry(
            sec,
            validator=self._validate_seed,
            on_validity_changed=self._on_validity_updated,
            initial_value=""
            if self.controller.genetic.seed is None
            else str(self.controller.genetic.seed),
            width=10,
        )
        self._seed_entry.grid(row=3, column=1, sticky="w", padx=(8, 0), pady=4)

    def _build_stopping_criteria_section(self) -> None:
        sec = ttk.LabelFrame(self, text="Criterios de Parada (RF-09)", padding=10)
        sec.pack(fill="x", pady=(0, 10))

        # Máximo de generaciones (Gmax)
        ttk.Label(sec, text="Máximo de generaciones (Gmax):").grid(
            row=0, column=0, sticky="w", pady=4
        )
        self._gmax_var = tk.IntVar(value=self.controller.stopping.max_generations)
        self._gmax_spin = ttk.Spinbox(
            sec,
            from_=1,
            to=100000,
            textvariable=self._gmax_var,
            width=10,
            command=self._on_stopping_changed,
        )
        self._gmax_spin.grid(row=0, column=1, sticky="w", padx=(8, 0), pady=4)
        self._gmax_spin.bind("<Return>", lambda e: self._on_stopping_changed())

        # Estancamiento toggle
        self._stag_enabled_var = tk.BooleanVar(value=self.controller.stopping.stagnation_enabled)
        self._stag_toggle = ttk.Checkbutton(
            sec,
            text="Parada por estancamiento de élite",
            variable=self._stag_enabled_var,
            bootstyle="round-toggle",
            command=self._on_stagnation_toggle,
        )
        self._stag_toggle.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 4))

        # Ventana K de generaciones
        ttk.Label(sec, text="Generaciones sin mejora (K):").grid(
            row=2, column=0, sticky="w", pady=4
        )
        self._stag_k_var = tk.IntVar(value=self.controller.stopping.stagnation_generations)
        self._stag_k_spin = ttk.Spinbox(
            sec,
            from_=2,
            to=1000,
            textvariable=self._stag_k_var,
            width=10,
            command=self._on_stopping_changed,
        )
        self._stag_k_spin.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=4)

        # Tolerancia épsilon
        ttk.Label(sec, text="Tolerancia de mejora (ε):").grid(row=3, column=0, sticky="w", pady=4)
        self._stag_eps_entry = ValidatedEntry(
            sec,
            validator=self._validate_positive_float,
            on_validity_changed=self._on_validity_updated,
            initial_value=str(self.controller.stopping.stagnation_epsilon),
            width=10,
        )
        self._stag_eps_entry.grid(row=3, column=1, sticky="w", padx=(8, 0), pady=4)

        self._on_stagnation_toggle()

    def _validate_prob(self, text: str) -> str | None:
        err = validate_number_format(text)
        if err:
            return err
        try:
            val = float(text.replace(",", "."))
            if not (0.0 <= val <= 1.0):
                return "Debe estar en el rango [0.0, 1.0]"
        except ValueError:
            return "Número decimal inválido"
        return None

    def _validate_seed(self, text: str) -> str | None:
        stripped = text.strip()
        if not stripped:
            return None
        return validate_number_format(stripped, integer_only=True)

    def _validate_positive_float(self, text: str) -> str | None:
        err = validate_number_format(text)
        if err:
            return err
        try:
            val = float(text.replace(",", "."))
            if val < 0.0:
                return "Debe ser mayor o igual a 0"
        except ValueError:
            return "Número decimal inválido"
        return None

    def _on_validity_updated(self, is_valid: bool) -> None:
        if (
            self._pc_entry.is_valid
            and self._pm_entry.is_valid
            and self._seed_entry.is_valid
            and self._stag_eps_entry.is_valid
        ):
            self._on_params_changed()
            self._on_stopping_changed()
            if self.on_validation_changed:
                self.on_validation_changed(True)
        else:
            if self.on_validation_changed:
                self.on_validation_changed(False)

    def _on_params_changed(self) -> None:
        with contextlib.suppress(Exception):
            np_val = int(self._np_var.get())
            if np_val >= 2:
                self.controller.set_population_size(np_val)

            if self._pc_entry.is_valid and self._pm_entry.is_valid:
                pc = float(self._pc_entry.value.replace(",", "."))
                pm = float(self._pm_entry.value.replace(",", "."))
                self.controller.set_probabilities(pc, pm)

            seed_str = self._seed_entry.value.strip()
            seed_val = int(seed_str) if seed_str else None
            self.controller.set_seed(seed_val)

    def _on_stagnation_toggle(self) -> None:
        enabled = self._stag_enabled_var.get()
        state = "normal" if enabled else "disabled"
        self._stag_k_spin.configure(state=state)
        self._stag_eps_entry.set_state(state)
        self._on_stopping_changed()

    def _on_stopping_changed(self) -> None:
        with contextlib.suppress(Exception):
            gmax = int(self._gmax_var.get())
            stag_on = self._stag_enabled_var.get()
            stag_k = int(self._stag_k_var.get())
            stag_eps = (
                float(self._stag_eps_entry.value.replace(",", "."))
                if self._stag_eps_entry.is_valid
                else 1e-6
            )
            self.controller.set_stopping_criteria(
                max_generations=gmax,
                stagnation_enabled=stag_on,
                stagnation_generations=stag_k,
                stagnation_epsilon=stag_eps,
            )

    def _on_model_changed(self) -> None:
        self._refresh_from_model()

    def _refresh_from_model(self) -> None:
        gen = self.controller.genetic
        stop = self.controller.stopping

        self._np_var.set(gen.population_size)
        self._pc_entry.value = str(gen.p_crossover)
        self._pm_entry.value = str(gen.p_mutation)
        self._seed_entry.value = "" if gen.seed is None else str(gen.seed)

        self._gmax_var.set(stop.max_generations)
        self._stag_enabled_var.set(stop.stagnation_enabled)
        self._stag_k_var.set(stop.stagnation_generations)
        self._stag_eps_entry.value = str(stop.stagnation_epsilon)
        self._on_stagnation_toggle()
