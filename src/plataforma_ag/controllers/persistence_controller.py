"""Controlador de persistencia e importación/exportación (RF-22..RF-24, TASK-052..054).

Coordina la lectura y escritura de configuraciones JSON, telemetría CSV
y reportes ejecutivos PDF, notificando actividad de E/S.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from pathlib import Path

from plataforma_ag.domain.config import ExperimentConfig
from plataforma_ag.services.config_store import export_config_to_json, import_config_from_json
from plataforma_ag.services.history import RunHistoryStore
from plataforma_ag.services.pdf_report import generate_pdf_report
from plataforma_ag.services.telemetry_csv import export_run_to_csv, import_run_from_csv


class PersistenceController:
    """Controlador para operaciones de persistencia y exportación documental."""

    def __init__(self) -> None:
        self._io_start_listeners: list[Callable[[], None]] = []
        self._io_finish_listeners: list[Callable[[], None]] = []

    def add_io_start_listener(self, cb: Callable[[], None]) -> None:
        self._io_start_listeners.append(cb)

    def add_io_finish_listener(self, cb: Callable[[], None]) -> None:
        self._io_finish_listeners.append(cb)

    def _notify_start(self) -> None:
        for cb in self._io_start_listeners:
            with contextlib.suppress(Exception):
                cb()

    def _notify_finish(self) -> None:
        for cb in self._io_finish_listeners:
            with contextlib.suppress(Exception):
                cb()

    def save_config_json(self, path: Path | str, config: ExperimentConfig) -> None:
        """Exporta la configuración del experimento a formato JSON."""
        self._notify_start()
        try:
            export_config_to_json(config, path)
        finally:
            self._notify_finish()

    def load_config_json(self, path: Path | str) -> ExperimentConfig:
        """Carga y valida una configuración de experimento desde formato JSON."""
        self._notify_start()
        try:
            return import_config_from_json(path)
        finally:
            self._notify_finish()

    def export_csv(self, path: Path | str, history: RunHistoryStore) -> None:
        """Exporta la serie completa de individuos de la corrida a CSV."""
        self._notify_start()
        try:
            export_run_to_csv(history, path)
        finally:
            self._notify_finish()

    def import_csv(self, path: Path | str) -> RunHistoryStore:
        """Importa y reconstruye el historial de una corrida desde un archivo CSV."""
        self._notify_start()
        try:
            return import_run_from_csv(path)
        finally:
            self._notify_finish()

    def generate_pdf_async(
        self,
        path: Path | str,
        config: ExperimentConfig,
        history: RunHistoryStore,
        chart_png_bytes: bytes,
        on_success: Callable[[Path], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        """Genera el informe técnico PDF en un hilo secundario sin bloquear la GUI."""
        self._notify_start()

        def _worker() -> None:
            try:
                out_path = generate_pdf_report(
                    config=config,
                    history=history,
                    chart_png_bytes=chart_png_bytes,
                    output_path=path,
                )
                on_success(out_path)
            except Exception as ex:
                on_error(ex)
            finally:
                self._notify_finish()

        threading.Thread(target=_worker, daemon=True).start()
