"""Composition root: único lugar que conecta capas (PROV-NF-01, RNF-04).

En el Sprint 1 solo arranca un shell mínimo; el ensamblado completo de vistas y
controladores llega con TASK-020.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence

from plataforma_ag import __version__

_INSTALL_HINT = (
    "No se encontró la biblioteca 'ttkbootstrap', necesaria para la interfaz.\n"
    "Instálala con:\n\n    pip install ttkbootstrap\n"
)

# Módulos no-GUI que deben importarse sin errores (verificación de arranque).
_CORE_MODULES: tuple[str, ...] = (
    "plataforma_ag.domain.enums",
    "plataforma_ag.domain.config",
    "plataforma_ag.domain.errors",
    "plataforma_ag.domain.events",
    "plataforma_ag.core.event_bus",
    "plataforma_ag.core.operators.base",
    "plataforma_ag.core.operators.registry",
    "plataforma_ag.core.problem.expression",
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="plataforma_ag",
        description="Plataforma de Simulación y Optimización con Algoritmos Genéticos",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Importa todos los módulos del núcleo y termina sin abrir la ventana.",
    )
    return parser.parse_args(argv)


def _check_imports() -> int:
    for name in _CORE_MODULES:
        importlib.import_module(name)
    print(f"plataforma_ag {__version__}: {len(_CORE_MODULES)} módulos del núcleo importados OK")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Arranca la aplicación. Devuelve el código de salida del proceso."""
    args = _parse_args(argv)
    if args.check:
        return _check_imports()

    try:
        from plataforma_ag.ui.views.main_window import MainWindow
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] == "ttkbootstrap":
            print(_INSTALL_HINT, file=sys.stderr)
            return 1
        raise

    window = MainWindow()
    window.run()
    return 0
