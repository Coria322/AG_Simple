"""Evaluador seguro de expresiones matemáticas (RF-02, RF-03, SUP-006).

Flujo: texto → normalización (``^`` → ``**``) → ``ast.parse`` → visitor de lista
blanca que, en la misma pasada, traduce cada nodo a un *closure* vectorizado de
numpy. Nunca se usa ``eval``/``exec``/``compile`` (regla innegociable n.º 1): el
resultado es una composición de funciones Python puras sobre arrays.

Gramática admitida:
    números · ``x1..xN`` · ``pi`` · ``e`` · ``+ - * / ^`` (y ``**``) · ``-x`` · ``( )``
    ``sin cos tan exp log sqrt abs`` (un argumento; trigonometría en **radianes**)

``log`` es logaritmo natural. ``-x1^2`` = ``-(x1^2)`` y ``^`` asocia por la derecha.
Errores numéricos (÷0, √ negativa, overflow) NO lanzan excepción: producen
``inf``/``nan`` que el motor penaliza (RNF-03, TASK-037).
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from plataforma_ag.domain import DTYPE
from plataforma_ag.domain.errors import ExpressionError

FloatArray = NDArray[np.float64]
_Fn = Callable[[FloatArray], FloatArray | np.float64]

ALLOWED_FUNCTIONS: Final[Mapping[str, np.ufunc]] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
}
"""Funciones admitidas (lista cerrada, RF-03 § Validaciones)."""

CONSTANTS: Final[Mapping[str, float]] = {"pi": float(np.pi), "e": float(np.e)}

DEGREE_IDENTIFIERS: Final[frozenset[str]] = frozenset(
    {"sind", "cosd", "tand", "deg", "degrees", "radians"}
)
"""Identificadores asociados a grados; se rechazan sin convertir (RF-03, TASK-005)."""

DEGREE_SYMBOLS: Final[tuple[str, ...]] = ("°", "º")

MAX_EXPRESSION_LENGTH: Final[int] = 2000
"""Tope defensivo de longitud para mantener el parseo acotado."""

_RADIANS_MESSAGE: Final[str] = (
    "Solo se admiten ángulos en radianes: usa sin/cos/tan con argumentos en radianes "
    "(por ejemplo sin(pi/2))"
)
_VARIABLE_RE: Final[re.Pattern[str]] = re.compile(r"x([1-9]\d*)")

_BINARY_OPS: Final[Mapping[type[ast.operator], np.ufunc]] = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.true_divide,
    ast.Pow: np.power,
}
_UNARY_OPS: Final[Mapping[type[ast.unaryop], np.ufunc]] = {
    ast.USub: np.negative,
    ast.UAdd: np.positive,
}
_FRIENDLY_NODE_NAMES: Final[Mapping[type[ast.AST], str]] = {
    ast.Compare: "comparaciones",
    ast.BoolOp: "operadores lógicos",
    ast.Attribute: "acceso a atributos",
    ast.Subscript: "índices o rebanadas",
    ast.Lambda: "funciones lambda",
    ast.IfExp: "expresiones condicionales",
    ast.List: "listas",
    ast.Tuple: "tuplas",
    ast.Dict: "diccionarios",
    ast.Set: "conjuntos",
    ast.JoinedStr: "cadenas de texto",
    ast.NamedExpr: "asignaciones",
}


# --- API pública -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompiledExpression:
    """Expresión validada y compilada una sola vez, lista para evaluar poblaciones."""

    source: str
    dimensions: int
    variables_used: frozenset[int]  # índices base 1 de las variables que aparecen
    _fn: _Fn = field(repr=False, compare=False)

    def evaluate(self, population: ArrayLike) -> FloatArray:
        """Evalúa una matriz ``(n_individuos, N)`` y devuelve un vector ``(n,)`` float64."""
        points = np.asarray(population, dtype=DTYPE)
        if points.ndim != 2 or points.shape[1] != self.dimensions:
            raise ValueError(
                f"Se esperaba una matriz (n, {self.dimensions}); se recibió {points.shape}"
            )
        with np.errstate(all="ignore"):  # RNF-03: inf/nan se penalizan después
            raw = self._fn(points)
        return np.broadcast_to(np.asarray(raw, dtype=DTYPE), (points.shape[0],)).copy()

    def evaluate_point(self, point: ArrayLike) -> float:
        """Evalúa un único punto ``(x1, …, xN)``."""
        vector = np.asarray(point, dtype=DTYPE).reshape(1, -1)
        return float(self.evaluate(vector)[0])


def compile_expression(text: str, dimensions: int) -> CompiledExpression:
    """Valida ``text`` contra la lista blanca y lo compila para ``dimensions`` variables.

    Raises:
        ExpressionError: si la expresión está vacía, es inválida, usa construcciones
            no permitidas, variables fuera de ``x1..xN`` o notación en grados.
    """
    if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions < 1:
        raise ExpressionError(f"El número de dimensiones debe ser un entero ≥ 1 ({dimensions!r})")
    if not isinstance(text, str):
        raise ExpressionError("La expresión debe ser texto")
    if not text.strip():
        raise ExpressionError("La expresión está vacía")
    if len(text) > MAX_EXPRESSION_LENGTH:
        raise ExpressionError(
            f"La expresión supera el máximo de {MAX_EXPRESSION_LENGTH} caracteres"
        )
    for symbol in DEGREE_SYMBOLS:
        if symbol in text:
            raise ExpressionError(_RADIANS_MESSAGE, position=text.index(symbol) + 1)

    translated, index_map = _normalize(text)
    tree = _parse(translated, index_map)
    compiler = _Compiler(dimensions, translated, index_map)
    fn = compiler.build(tree.body)
    return CompiledExpression(
        source=text,
        dimensions=dimensions,
        variables_used=frozenset(compiler.variables_used),
        _fn=fn,
    )


def validate_expression(text: str, dimensions: int) -> None:
    """Solo valida (para la GUI): lanza :class:`ExpressionError` si es inválida."""
    compile_expression(text, dimensions)


# --- Implementación ----------------------------------------------------------


def _normalize(text: str) -> tuple[str, list[int]]:
    """Traduce ``^`` → ``**`` y saltos de línea → espacio.

    Devuelve el texto traducido y, para cada carácter traducido, su índice (base 0)
    en el texto original, para reportar posiciones correctas.
    """
    out: list[str] = []
    index_map: list[int] = []
    for i, ch in enumerate(text):
        if ch == "^":
            out.append("**")
            index_map.extend((i, i))
        elif ch in "\r\n\t":
            out.append(" ")
            index_map.append(i)
        else:
            out.append(ch)
            index_map.append(i)
    index_map.append(len(text))  # posición "fin de texto"
    return "".join(out), index_map


def _original_position(index_map: list[int], translated_index: int | None) -> int | None:
    if translated_index is None:
        return None
    clamped = min(max(translated_index, 0), len(index_map) - 1)
    return index_map[clamped] + 1  # base 1 para el usuario


def _parse(translated: str, index_map: list[int]) -> ast.Expression:
    try:
        return ast.parse(translated, mode="eval")
    except SyntaxError as exc:
        offset = (exc.offset - 1) if exc.offset else None
        reason = _syntax_reason(exc)
        raise ExpressionError(
            f"Sintaxis inválida: {reason}", position=_original_position(index_map, offset)
        ) from None
    except (RecursionError, MemoryError, ValueError):
        raise ExpressionError(
            "La expresión es demasiado compleja o contiene caracteres inválidos"
        ) from None


def _syntax_reason(exc: SyntaxError) -> str:
    msg = (exc.msg or "").lower()
    if "never closed" in msg or "unexpected eof" in msg:
        return "falta cerrar un paréntesis"
    if "unmatched" in msg or "does not match" in msg:
        return "paréntesis de cierre sin apertura"
    if "invalid decimal literal" in msg:
        return "número mal formado (¿falta un operador, p. ej. 2*x1?)"
    if "perhaps you forgot a comma" in msg:
        return "falta un operador entre dos términos"
    return "revisa operadores y paréntesis"


class _Compiler:
    """Visitor de lista blanca que valida y compila en una sola pasada."""

    def __init__(self, dimensions: int, translated: str, index_map: list[int]) -> None:
        self.dimensions = dimensions
        self.variables_used: set[int] = set()
        self._encoded = translated.encode("utf-8")
        self._index_map = index_map

    # -- utilidades --------------------------------------------------------

    def _position(self, node: ast.AST) -> int | None:
        col = getattr(node, "col_offset", None)
        if col is None:
            return None
        char_index = len(self._encoded[:col].decode("utf-8", errors="ignore"))
        return _original_position(self._index_map, char_index)

    def _error(self, message: str, node: ast.AST) -> ExpressionError:
        return ExpressionError(message, position=self._position(node))

    # -- despacho ----------------------------------------------------------

    def build(self, node: ast.AST) -> _Fn:
        if isinstance(node, ast.BinOp):
            return self._binop(node)
        if isinstance(node, ast.UnaryOp):
            return self._unary(node)
        if isinstance(node, ast.Call):
            return self._call(node)
        if isinstance(node, ast.Name):
            return self._name(node)
        if isinstance(node, ast.Constant):
            return self._constant(node)
        what = _FRIENDLY_NODE_NAMES.get(type(node), type(node).__name__)
        raise self._error(f"Construcción no permitida: {what}", node)

    # -- nodos -------------------------------------------------------------

    def _binop(self, node: ast.BinOp) -> _Fn:
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise self._error(
                f"Operador no permitido: {type(node.op).__name__}. "
                "Usa + - * / ^",
                node,
            )
        left, right = self.build(node.left), self.build(node.right)

        def binop(x: FloatArray) -> FloatArray:
            return op(left(x), right(x))  # type: ignore[no-any-return]

        return binop

    def _unary(self, node: ast.UnaryOp) -> _Fn:
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise self._error(f"Operador unario no permitido: {type(node.op).__name__}", node)
        operand = self.build(node.operand)

        def unary(x: FloatArray) -> FloatArray:
            return op(operand(x))  # type: ignore[no-any-return]

        return unary

    def _call(self, node: ast.Call) -> _Fn:
        if not isinstance(node.func, ast.Name):
            raise self._error("Solo se pueden llamar funciones por su nombre", node)
        name = node.func.id
        if name in DEGREE_IDENTIFIERS:
            raise self._error(_RADIANS_MESSAGE, node)
        func = ALLOWED_FUNCTIONS.get(name)
        if func is None:
            allowed = ", ".join(ALLOWED_FUNCTIONS)
            raise self._error(
                f"Función no permitida '{name}'. Funciones admitidas: {allowed}", node
            )
        if node.keywords or len(node.args) != 1 or isinstance(node.args[0], ast.Starred):
            raise self._error(f"La función '{name}' recibe exactamente un argumento", node)
        arg = self.build(node.args[0])

        def call(x: FloatArray) -> FloatArray:
            return func(arg(x))  # type: ignore[no-any-return]

        return call

    def _name(self, node: ast.Name) -> _Fn:
        name = node.id
        if name in DEGREE_IDENTIFIERS:
            raise self._error(_RADIANS_MESSAGE, node)
        match = _VARIABLE_RE.fullmatch(name)
        if match:
            index = int(match.group(1))
            if index > self.dimensions:
                raise self._error(
                    f"La variable '{name}' no existe: con N={self.dimensions} "
                    f"solo se admiten x1..x{self.dimensions}",
                    node,
                )
            self.variables_used.add(index)
            column = index - 1

            def variable(x: FloatArray) -> FloatArray:
                return x[:, column]

            return variable
        if name in CONSTANTS:
            return self._const_fn(CONSTANTS[name])
        if name in ALLOWED_FUNCTIONS:
            raise self._error(f"'{name}' es una función: úsala como {name}(...)", node)
        raise self._error(
            f"Identificador desconocido '{name}'. Usa x1..x{self.dimensions}, pi o e", node
        )

    def _constant(self, node: ast.Constant) -> _Fn:
        value = node.value
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise self._error("Solo se admiten constantes numéricas reales", node)
        try:
            number = float(value)
        except OverflowError:
            raise self._error("Constante numérica demasiado grande", node) from None
        return self._const_fn(number)

    @staticmethod
    def _const_fn(value: float) -> _Fn:
        constant = DTYPE(value)

        def const(_: FloatArray) -> np.float64:
            return constant

        return const
