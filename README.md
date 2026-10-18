# Plataforma AG — Simulación y Optimización con Algoritmos Genéticos

Aplicación de escritorio (Python + ttkbootstrap) para modelar, ejecutar y analizar
Algoritmos Genéticos en R^N. Documentación agéntica en
[`plataforma_ag_agentic_docs/plataforma_ag`](plataforma_ag_agentic_docs/plataforma_ag/docs/AGENTS.md).

## Instalación

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Uso

```powershell
.\.venv\Scripts\python.exe -m plataforma_ag          # abre la GUI
.\.venv\Scripts\python.exe -m plataforma_ag --check  # verifica importaciones sin GUI
```

## Calidad

```powershell
.\.venv\Scripts\python.exe -m pytest        # pruebas (incluye contrato de capas)
.\.venv\Scripts\lint-imports.exe            # import-linter
.\.venv\Scripts\ruff.exe check src tests
.\.venv\Scripts\mypy.exe
```

## Arquitectura

`app → ui → controllers → (core | services) → domain`. `core`, `domain` y `services`
no importan `tkinter`, `ttkbootstrap`, `ui` ni `controllers`.

Decisiones sobre ambigüedades del ERS: ver [`docs/DECISIONS.md`](docs/DECISIONS.md).
