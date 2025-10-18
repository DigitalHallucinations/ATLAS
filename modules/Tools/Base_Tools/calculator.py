"""Precision calculator tool with strict expression validation."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from decimal import (
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Context,
    localcontext,
    ROUND_CEILING,
    ROUND_FLOOR,
)
from math import e, pi, tau
from typing import Any, Dict, Mapping, Optional


class CalculatorError(RuntimeError):
    """Base class for calculator failures."""


class ExpressionSyntaxError(CalculatorError):
    """Raised when the input expression cannot be parsed."""


class ExpressionValidationError(CalculatorError):
    """Raised when the expression uses unsupported constructs."""


class UnitConversionError(CalculatorError):
    """Raised when unit conversions cannot be resolved."""


class EvaluationError(CalculatorError):
    """Raised when evaluating the expression fails."""


@dataclass(frozen=True)
class CalculatorResult:
    """Structured response returned by :class:`Calculator`."""

    result: str
    value: Decimal
    unit: Optional[str]
    precision: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "result": self.result,
            "value": str(self.value),
            "unit": self.unit,
            "precision": self.precision,
        }


_ALLOWED_BINOPS = {
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
}

_ALLOWED_UNARYOPS = {ast.UAdd, ast.USub}
_ALLOWED_CALLS = {
    "abs",
    "ceil",
    "floor",
    "round",
    "sqrt",
    "exp",
    "ln",
    "log",
    "log10",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "min",
    "max",
}

_UNIT_FACTORS = {
    "m": Decimal("1"),
    "cm": Decimal("0.01"),
    "mm": Decimal("0.001"),
    "km": Decimal("1000"),
    "inch": Decimal("0.0254"),
    "ft": Decimal("0.3048"),
    "yd": Decimal("0.9144"),
    "mile": Decimal("1609.34"),
    "g": Decimal("0.001"),
    "kg": Decimal("1"),
    "lb": Decimal("0.45359237"),
    "oz": Decimal("0.0283495"),
    "l": Decimal("0.001"),
    "ml": Decimal("0.000001"),
    "gal": Decimal("0.00378541"),
    "c": Decimal("1"),
    "f": Decimal("1"),
    "s": Decimal("1"),
    "min": Decimal("60"),
    "h": Decimal("3600"),
}


def _decimal_math(func):
    def wrapper(*values: Decimal) -> Decimal:
        floats = [float(value) for value in values]
        result = func(*floats)
        return Decimal(str(result))

    return wrapper


class _DecimalEvaluator(ast.NodeVisitor):
    """Evaluate an expression AST using :class:`Decimal` semantics."""

    def __init__(
        self,
        *,
        precision: int,
        variables: Mapping[str, Decimal],
    ) -> None:
        self._context = Context(prec=max(precision, 1))
        self._variables = variables
        self._functions: Mapping[str, Any] = {
            "abs": lambda x: x.copy_abs(),
            "ceil": lambda x: x.to_integral_exact(rounding=ROUND_CEILING),
            "floor": lambda x: x.to_integral_exact(rounding=ROUND_FLOOR),
            "round": self._round,
            "sqrt": lambda x: x.sqrt(),
            "exp": lambda x: x.exp(),
            "ln": lambda x: x.ln(),
            "log": lambda x, base=Decimal("10"): x.ln() / Decimal(base).ln(),
            "log10": lambda x: x.log10(),
            "sin": _decimal_math(math.sin),
            "cos": _decimal_math(math.cos),
            "tan": _decimal_math(math.tan),
            "asin": _decimal_math(math.asin),
            "acos": _decimal_math(math.acos),
            "atan": _decimal_math(math.atan),
            "sinh": _decimal_math(math.sinh),
            "cosh": _decimal_math(math.cosh),
            "tanh": _decimal_math(math.tanh),
            "min": lambda *values: min(values),
            "max": lambda *values: max(values),
        }

    def evaluate(self, node: ast.AST) -> Decimal:
        with localcontext(self._context):
            return self.visit(node)

    def visit_Expression(self, node: ast.Expression) -> Decimal:  # type: ignore[override]
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> Decimal:  # type: ignore[override]
        if type(node.op) not in _ALLOWED_BINOPS:
            raise ExpressionValidationError(f"Operator '{type(node.op).__name__}' is not supported.")
        left = self.visit(node.left)
        right = self.visit(node.right)
        try:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                return left.__pow__(right)
        except (DivisionByZero, InvalidOperation) as exc:
            raise EvaluationError(str(exc)) from exc
        raise ExpressionValidationError("Unsupported binary operation.")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Decimal:  # type: ignore[override]
        if type(node.op) not in _ALLOWED_UNARYOPS:
            raise ExpressionValidationError("Unsupported unary operator.")
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ExpressionValidationError("Unsupported unary operation.")

    def visit_Call(self, node: ast.Call) -> Decimal:  # type: ignore[override]
        if not isinstance(node.func, ast.Name):
            raise ExpressionValidationError("Only simple function names are allowed.")
        name = node.func.id
        if name not in _ALLOWED_CALLS:
            raise ExpressionValidationError(f"Function '{name}' is not supported.")
        function = self._functions.get(name)
        if function is None:
            raise ExpressionValidationError(f"Function '{name}' is unavailable in this context.")
        args = [self.visit(arg) for arg in node.args]
        if node.keywords:
            raise ExpressionValidationError("Keyword arguments are not supported.")
        try:
            return function(*args)
        except (TypeError, InvalidOperation, DivisionByZero) as exc:
            raise EvaluationError(str(exc)) from exc

    def visit_Name(self, node: ast.Name) -> Decimal:  # type: ignore[override]
        if node.id not in self._variables:
            raise ExpressionValidationError(f"Unknown identifier '{node.id}'.")
        return self._variables[node.id]

    def visit_Constant(self, node: ast.Constant) -> Decimal:  # type: ignore[override]
        value = node.value
        if isinstance(value, (int, float, complex)):
            if isinstance(value, complex):
                raise ExpressionValidationError("Complex numbers are not supported.")
            return Decimal(str(value))
        if isinstance(value, str):
            raise ExpressionValidationError("String literals are not allowed.")
        if value is None:
            raise ExpressionValidationError("None is not a valid literal in calculator expressions.")
        raise ExpressionValidationError(f"Unsupported literal type: {type(value).__name__}.")

    def generic_visit(self, node: ast.AST) -> Decimal:  # type: ignore[override]
        raise ExpressionValidationError(f"Unsupported expression element: {type(node).__name__}.")

    def _round(self, value: Decimal, ndigits: Optional[Decimal] = None) -> Decimal:
        if ndigits is None:
            return value.quantize(Decimal("1"))
        try:
            digits = int(ndigits)
        except (ValueError, TypeError):
            raise ExpressionValidationError("round() expects an integer number of digits.")
        exponent = Decimal("1").scaleb(-digits)
        return value.quantize(exponent)


class Calculator:
    """Evaluate sanitized mathematical expressions using :class:`Decimal`."""

    def __init__(
        self,
        *,
        precision: int = 28,
        max_ast_nodes: int = 256,
        unit_factors: Optional[Mapping[str, Decimal]] = None,
    ) -> None:
        if precision <= 0:
            raise ValueError("precision must be positive")
        self._default_precision = int(precision)
        self._max_nodes = max(8, int(max_ast_nodes))
        merged_units = dict(_UNIT_FACTORS)
        if unit_factors:
            for key, value in unit_factors.items():
                merged_units[str(key).lower()] = Decimal(str(value))
        self._units: Mapping[str, Decimal] = merged_units
        self._constants: Mapping[str, Decimal] = {
            "pi": Decimal(str(pi)),
            "tau": Decimal(str(tau)),
            "e": Decimal(str(e)),
        }

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]]) -> "Calculator":
        config = dict(config or {})
        precision = int(config.get("precision", 28))
        max_nodes = int(config.get("max_ast_nodes", 256))
        custom_units = config.get("unit_factors")
        if isinstance(custom_units, Mapping):
            factors = {str(key): Decimal(str(value)) for key, value in custom_units.items()}
        else:
            factors = None
        return cls(precision=precision, max_ast_nodes=max_nodes, unit_factors=factors)

    def evaluate(
        self,
        expression: str,
        *,
        precision: Optional[int] = None,
        variables: Optional[Mapping[str, Any]] = None,
        input_unit: Optional[str] = None,
        output_unit: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not isinstance(expression, str) or not expression.strip():
            raise CalculatorError("Expression must be a non-empty string.")

        precision_value = int(precision or self._default_precision)
        tree = self._parse_expression(expression)
        self._validate_tree(tree)

        values = self._prepare_variables(variables)
        evaluator = _DecimalEvaluator(precision=precision_value, variables=values)
        try:
            decimal_value = evaluator.evaluate(tree)
        except (ExpressionValidationError, EvaluationError):
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise EvaluationError(str(exc)) from exc

        unit = input_unit.lower() if input_unit else None
        target = output_unit.lower() if output_unit else None
        if unit or target:
            decimal_value, unit = self._convert_units(decimal_value, unit, target)

        formatted = self._format_decimal(decimal_value)
        return CalculatorResult(
            result=formatted,
            value=decimal_value,
            unit=unit,
            precision=precision_value,
        ).as_dict()

    def _parse_expression(self, expression: str) -> ast.Expression:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ExpressionSyntaxError(str(exc)) from exc
        if not isinstance(tree, ast.Expression):
            raise ExpressionSyntaxError("Only single expressions are allowed.")
        return tree

    def _validate_tree(self, tree: ast.Expression) -> None:
        nodes = list(ast.walk(tree))
        if len(nodes) > self._max_nodes:
            raise ExpressionValidationError(
                f"Expression is too complex (node budget {self._max_nodes} exceeded)."
            )
        for node in nodes:
            if isinstance(node, (ast.Load, ast.Expression)):
                continue
            if isinstance(node, (ast.Module, ast.Assign, ast.AugAssign, ast.Attribute, ast.Subscript, ast.Lambda)):
                raise ExpressionValidationError("Only arithmetic expressions are allowed.")

    def _prepare_variables(self, variables: Optional[Mapping[str, Any]]) -> Mapping[str, Decimal]:
        prepared: Dict[str, Decimal] = dict(self._constants)
        if variables:
            for key, value in variables.items():
                if not isinstance(key, str):
                    raise ExpressionValidationError("Variable names must be strings.")
                prepared[key] = Decimal(str(value))
        return prepared

    def _convert_units(
        self,
        value: Decimal,
        input_unit: Optional[str],
        output_unit: Optional[str],
    ) -> tuple[Decimal, Optional[str]]:
        if input_unit and input_unit not in self._units:
            raise UnitConversionError(f"Unknown source unit '{input_unit}'.")
        if output_unit and output_unit not in self._units:
            raise UnitConversionError(f"Unknown target unit '{output_unit}'.")

        if not input_unit and not output_unit:
            return value, None
        if input_unit and not output_unit:
            factor = self._units[input_unit]
            return value * factor, input_unit
        if not input_unit and output_unit:
            factor = self._units[output_unit]
            return value / factor, output_unit

        source_factor = self._units[input_unit]
        target_factor = self._units[output_unit]
        converted = value * source_factor / target_factor
        return converted, output_unit

    def _format_decimal(self, value: Decimal) -> str:
        normalized = value.normalize()
        if normalized == normalized.to_integral():
            return str(normalized.to_integral())
        text = format(normalized, "f").rstrip("0").rstrip(".")
        return text or "0"
