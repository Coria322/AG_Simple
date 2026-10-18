"""Modelo de dominio: enums, dataclasses, errores y eventos (sin dependencias de GUI).

Única dependencia externa permitida: numpy, para declarar ``DTYPE`` (PROV-NF-02).
"""

import numpy as np

DTYPE: type[np.float64] = np.float64
"""Tipo numérico de todo el cálculo: IEEE 754 doble precisión (PROV-NF-02)."""

__all__ = ["DTYPE"]
