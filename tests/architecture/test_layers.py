"""TASK-001/034 — Contrato de capas con import-linter (RNF-04)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.architecture

ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = ROOT / "src" / "plataforma_ag" / "core"


def _lint_imports() -> subprocess.CompletedProcess[str]:
    exe = shutil.which("lint-imports", path=str(Path(sys.executable).parent))
    if exe is None:
        pytest.skip("import-linter no está instalado")
    return subprocess.run(
        [exe, "--no-cache"], cwd=ROOT, capture_output=True, text=True, timeout=120, check=False
    )


def test_contrato_de_capas_se_cumple() -> None:
    result = _lint_imports()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "BROKEN" not in result.stdout


@pytest.mark.parametrize("forbidden", ["tkinter", "ttkbootstrap", "plataforma_ag.ui"])
def test_core_importando_gui_rompe_el_contrato(forbidden: str) -> None:
    probe = CORE_DIR / "_probe_violacion_capas.py"
    probe.write_text(f"import {forbidden}  # sonda temporal de prueba\n", encoding="utf-8")
    try:
        result = _lint_imports()
    finally:
        probe.unlink(missing_ok=True)
    assert result.returncode != 0, "import-linter debió detectar la violación"
    assert "BROKEN" in result.stdout
