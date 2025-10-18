from decimal import Decimal
import datetime
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

yaml_stub = ModuleType("yaml")
yaml_stub.safe_load = lambda *_args, **_kwargs: {}
yaml_stub.dump = lambda *_args, **_kwargs: None
sys.modules.setdefault("yaml", yaml_stub)

dotenv_stub = ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda *_args, **_kwargs: None
dotenv_stub.set_key = lambda *_args, **_kwargs: None
dotenv_stub.find_dotenv = lambda *_args, **_kwargs: ""
sys.modules.setdefault("dotenv", dotenv_stub)


class _StubTimezone(datetime.tzinfo):
    def __init__(self, name: str) -> None:
        self._name = name

    def utcoffset(self, _dt):
        return datetime.timedelta(0)

    def dst(self, _dt):
        return datetime.timedelta(0)

    def tzname(self, _dt):
        return self._name


pytz_stub = ModuleType("pytz")
pytz_stub.timezone = lambda name: _StubTimezone(name)
sys.modules.setdefault("pytz", pytz_stub)

MODULE_PATH = Path(__file__).resolve().parents[1] / "modules" / "Tools" / "Base_Tools" / "calculator.py"
spec = importlib.util.spec_from_file_location("calculator_test_module", MODULE_PATH)
calculator_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = calculator_module
spec.loader.exec_module(calculator_module)

Calculator = calculator_module.Calculator
ExpressionValidationError = calculator_module.ExpressionValidationError
UnitConversionError = calculator_module.UnitConversionError


def test_simple_expression():
    tool = Calculator()
    result = tool.evaluate("2 + 3 * 4")
    assert result["result"] == "14"
    assert Decimal(result["value"]) == Decimal("14")


def test_precision_override():
    tool = Calculator(precision=12)
    result = tool.evaluate("1 / 3", precision=6)
    assert result["precision"] == 6
    assert result["result"].startswith("0.333333")


def test_trigonometry_and_variables():
    tool = Calculator()
    result = tool.evaluate("sin(pi / 2) + x", variables={"x": 1})
    assert abs(float(result["result"]) - 2.0) < 1e-6


def test_unit_conversion():
    tool = Calculator()
    result = tool.evaluate("5", input_unit="km", output_unit="m")
    assert result["unit"] == "m"
    assert Decimal(result["value"]) == Decimal("5000")


def test_invalid_expression_rejected():
    tool = Calculator()
    with pytest.raises(ExpressionValidationError):
        tool.evaluate("(lambda x: x)(2)")


def test_unknown_unit_raises():
    tool = Calculator()
    with pytest.raises(UnitConversionError):
        tool.evaluate("10", input_unit="parsec")
