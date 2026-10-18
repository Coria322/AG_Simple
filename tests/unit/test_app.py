"""TASK-001 — Arranque del paquete y mensaje de dependencia ausente."""

from __future__ import annotations

import subprocess
import sys

import pytest

from plataforma_ag import app


def test_python_m_plataforma_ag_arranca_sin_errores_de_importacion() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "plataforma_ag", "--check"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "importados OK" in result.stdout


def test_version() -> None:
    with pytest.raises(SystemExit) as info:
        app.main(["--version"])
    assert info.value.code == 0


def test_sin_ttkbootstrap_muestra_instruccion_y_termina_limpio(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in list(sys.modules):
        if name == "ttkbootstrap" or name.startswith(("ttkbootstrap.", "plataforma_ag.ui")):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "ttkbootstrap", None)  # fuerza ModuleNotFoundError

    assert app.main([]) == 1
    assert "pip install ttkbootstrap" in capsys.readouterr().err
