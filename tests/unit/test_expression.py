"""TASK-028 — Pruebas del evaluador de expresiones y radianes (RF-02, RF-03)."""

from __future__ import annotations

import numpy as np
import pytest

from plataforma_ag.core.problem import expression as expr_mod
from plataforma_ag.core.problem.expression import (
    DEGREE_IDENTIFIERS,
    compile_expression,
    validate_expression,
)
from plataforma_ag.domain.errors import ExpressionError

# --- RF-02 Escenario 1: evaluación correcta ----------------------------------


def test_suma_de_cuadrados_en_3_4_es_25() -> None:
    f = compile_expression("x1^2 + x2^2", dimensions=2)
    assert f.evaluate_point((3, 4)) == 25.0


def test_evaluacion_vectorizada_devuelve_float64_por_individuo() -> None:
    f = compile_expression("x1^2 + x2^2", dimensions=2)
    pop = np.array([[0, 0], [1, 1], [3, 4]])
    out = f.evaluate(pop)
    assert out.dtype == np.float64
    assert out.shape == (3,)
    np.testing.assert_array_equal(out, [0.0, 2.0, 25.0])


def test_expresion_constante_se_difunde_a_toda_la_poblacion() -> None:
    f = compile_expression("pi + 1", dimensions=3)
    out = f.evaluate(np.zeros((4, 3)))
    np.testing.assert_allclose(out, np.full(4, np.pi + 1))


@pytest.mark.parametrize(
    ("text", "point", "expected"),
    [
        ("-x1^2", (3.0,), -9.0),  # -(x1^2), no (-x1)^2
        ("2^3^2", (0.0,), 512.0),  # asociatividad derecha
        ("x1**2", (3.0,), 9.0),  # ** también es válido
        ("2^-1", (0.0,), 0.5),
        ("(x1 + 1) * 2", (1.0,), 4.0),
        ("10 / 4 - 1", (0.0,), 1.5),
        ("+x1", (7.0,), 7.0),
        ("e", (0.0,), np.e),
        ("log(e)", (0.0,), 1.0),
        ("exp(0) + sqrt(16) + abs(-2)", (0.0,), 7.0),
        ("1e3 * x1", (2.0,), 2000.0),
    ],
)
def test_precedencia_y_funciones(text: str, point: tuple[float, ...], expected: float) -> None:
    assert compile_expression(text, dimensions=len(point)).evaluate_point(point) == pytest.approx(
        expected, rel=1e-15
    )


def test_rastrigin_escrito_como_expresion_libre() -> None:
    text = "20 + x1^2 - 10*cos(2*pi*x1) + x2^2 - 10*cos(2*pi*x2)"
    f = compile_expression(text, dimensions=2)
    assert f.evaluate_point((0.0, 0.0)) == pytest.approx(0.0, abs=1e-12)
    assert f.variables_used == frozenset({1, 2})


def test_saltos_de_linea_se_toleran() -> None:
    assert compile_expression("x1 +\n x2", 2).evaluate_point((1, 2)) == 3.0


# --- RF-02 Escenario 2: expresión maliciosa o inválida ------------------------


@pytest.mark.parametrize(
    "text",
    [
        "__import__('os').system('x')",
        "__import__('os')",
        "open('archivo.txt')",
        "x1.__class__",
        "(lambda: 1)()",
        "[x1 for x1 in range(3)]",
        "x1 if x1 > 0 else 0",
        "x1 > 0",
        "x1 and x2",
        "'texto'",
        "x1[0]",
        "eval('1+1')",
        "sin(x=1)",
        "sin(x1, x2)",
        "sin()",
        "np.sin(x1)",
        "(y := 3)",
        "True + 1",
        "1j",
        "x1 % 2",
        "x1 // 2",
    ],
)
def test_construcciones_prohibidas_lanzan_expression_error(text: str) -> None:
    with pytest.raises(ExpressionError):
        compile_expression(text, dimensions=2)


def test_codigo_malicioso_no_se_ejecuta(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    called: list[str] = []
    monkeypatch.setattr(os, "system", lambda cmd: called.append(cmd) or 0)
    with pytest.raises(ExpressionError):
        compile_expression("__import__('os').system('echo hackeado')", dimensions=1)
    assert called == []


def test_el_modulo_no_usa_eval_exec_ni_compile() -> None:
    import inspect
    import re

    source = inspect.getsource(expr_mod)
    pattern = re.compile(r"(?<![\w.])(eval|exec|compile|__import__)\s*\(")
    assert pattern.findall(source) == []


def test_variable_fuera_de_n_se_rechaza_con_mensaje() -> None:
    with pytest.raises(ExpressionError, match=r"x3.*N=2"):
        compile_expression("x3 + 1", dimensions=2)


@pytest.mark.parametrize("name", ["x0", "x01", "X1", "y", "xx1", "Sin"])
def test_identificadores_invalidos(name: str) -> None:
    with pytest.raises(ExpressionError, match="desconocido"):
        compile_expression(f"{name} + 1", dimensions=2)


def test_funcion_desconocida_lista_las_admitidas() -> None:
    with pytest.raises(ExpressionError, match=r"Función no permitida 'cosh'.*sin, cos"):
        compile_expression("cosh(x1)", dimensions=1)


def test_funcion_usada_como_variable() -> None:
    with pytest.raises(ExpressionError, match=r"sin\(\.\.\.\)"):
        compile_expression("sin + 1", dimensions=1)


# --- Casos borde de RF-02 -----------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_cadena_vacia_es_error(text: str) -> None:
    with pytest.raises(ExpressionError, match="vacía"):
        compile_expression(text, dimensions=1)


def test_parentesis_sin_cerrar_reporta_posicion() -> None:
    with pytest.raises(ExpressionError, match="paréntesis") as info:
        compile_expression("(x1 + 2", dimensions=1)
    assert info.value.position is not None


def test_parentesis_de_cierre_sobrante() -> None:
    with pytest.raises(ExpressionError, match="paréntesis"):
        compile_expression("x1 + 2)", dimensions=1)


def test_posicion_se_refiere_al_texto_original_tras_traducir_circunflejo() -> None:
    # 'x1^2 + x2^2 + cosh(x1)': cosh empieza en la columna 15 del texto original.
    with pytest.raises(ExpressionError) as info:
        compile_expression("x1^2 + x2^2 + cosh(x1)", dimensions=2)
    assert info.value.position == 15


def test_dimensiones_invalidas() -> None:
    for bad in (0, -1, True, 2.0):
        with pytest.raises(ExpressionError):
            compile_expression("x1", dimensions=bad)  # type: ignore[arg-type]


def test_expresion_demasiado_larga() -> None:
    with pytest.raises(ExpressionError, match="máximo"):
        compile_expression("+".join(["x1"] * 1500), dimensions=1)


def test_constante_entera_gigante() -> None:
    with pytest.raises(ExpressionError, match="grande"):
        compile_expression("1" + "0" * 400, dimensions=1)


def test_errores_numericos_no_lanzan_y_devuelven_no_finitos() -> None:
    f = compile_expression("1/x1 + sqrt(x2) + exp(x3)", dimensions=3)
    out = f.evaluate(np.array([[0.0, 1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 1e6]]))
    assert not np.isfinite(out).any()


def test_matriz_con_forma_incorrecta() -> None:
    f = compile_expression("x1 + x2", dimensions=2)
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        f.evaluate(np.zeros((5, 3)))


def test_validate_expression() -> None:
    validate_expression("sin(x1)", 1)
    with pytest.raises(ExpressionError):
        validate_expression("sin(x2)", 1)


# --- RF-03 Radianes (TASK-005) -------------------------------------------------


def test_sin_pi_medios_es_exactamente_uno() -> None:
    assert compile_expression("sin(pi/2)", dimensions=1).evaluate_point((0.0,)) == 1.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [("cos(pi)", -1.0), ("sin(0)", 0.0), ("tan(pi/4)", 1.0), ("cos(x1)", np.cos(1.0))],
)
def test_trigonometria_en_radianes(text: str, expected: float) -> None:
    assert compile_expression(text, 1).evaluate_point((1.0,)) == pytest.approx(expected, abs=1e-15)


def test_tan_pi_medios_es_grande_y_finito() -> None:
    value = compile_expression("tan(pi/2)", 1).evaluate_point((0.0,))
    assert np.isfinite(value) and abs(value) > 1e15


@pytest.mark.parametrize("text", ["sind(90)", "cosd(x1)", "tand(45)", "sin(deg(x1))",
                                  "degrees(x1)", "radians(90)", "sin(90°)", "sin(90º)", "deg"])
def test_construcciones_en_grados_se_rechazan(text: str) -> None:
    with pytest.raises(ExpressionError, match="radianes"):
        compile_expression(text, dimensions=1)


def test_lista_de_rechazo_de_grados_es_cerrada() -> None:
    assert frozenset({"sind", "cosd", "tand", "deg", "degrees", "radians"}) == DEGREE_IDENTIFIERS
