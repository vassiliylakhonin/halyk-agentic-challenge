"""Deterministic arithmetic check.

The model states each computation as a literal expression plus its claimed value.
We recompute the expression in Python and compare. A language model can be wrong
about 250000000 * 0.145 * 90 / 365; the interpreter cannot. Scoring counts
computational accuracy separately, so this is the cheapest point of leverage in
the whole pipeline.
"""

from __future__ import annotations

import ast
import math
import re

from .schema import Step

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub,
    ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    ast.Call, ast.Name, ast.Load, ast.Tuple,
)

_ALLOWED_FUNCS = {
    "round": round, "abs": abs, "min": min, "max": max,
    "sqrt": math.sqrt, "floor": math.floor, "ceil": math.ceil, "log": math.log,
    "exp": math.exp, "pow": pow,
}

_CLEAN = re.compile(r"[   _']")
_PERCENT = re.compile(r"(?<=[\d)])\s*%")


def normalize(expr: str) -> str:
    """Strip formatting a model tends to leave in: spaces, NBSP, thousands marks,
    trailing percent signs (turned into /100), and unicode operators."""
    s = expr.strip()
    s = s.replace("×", "*").replace("·", "*").replace("÷", "/").replace("−", "-")
    s = s.replace("^", "**")
    s = _CLEAN.sub("", s)
    s = _PERCENT.sub("/100", s)
    return s


def safe_eval(expr: str) -> float:
    tree = ast.parse(normalize(expr), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
                raise ValueError("only round/abs/min/max/sqrt/floor/ceil/log/exp/pow allowed")
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_FUNCS:
            raise ValueError(f"variables are not allowed: {node.id}")
    return float(eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, dict(_ALLOWED_FUNCS)))


def check_steps(steps: list[Step], rel_tol: float = 1e-6, abs_tol: float = 0.01
                ) -> tuple[bool, list[str], list[Step]]:
    """Returns (all_ok, notes, corrected_steps). Values that recompute differently
    are replaced with the computed value and flagged."""
    ok = True
    notes: list[str] = []
    fixed: list[Step] = []
    for i, st in enumerate(steps, start=1):
        try:
            got = safe_eval(st.expression)
        except Exception as e:
            ok = False
            notes.append(f"step {i} ({st.label}): expression not machine-checkable "
                         f"({type(e).__name__}: {e}); expression was {st.expression!r}")
            fixed.append(st)
            continue
        if not math.isclose(got, st.value, rel_tol=rel_tol, abs_tol=abs_tol):
            ok = False
            notes.append(f"step {i} ({st.label}): stated {st.value}, recomputed {got} "
                         f"from {st.expression!r}")
            fixed.append(Step(label=st.label, expression=st.expression, value=got))
        else:
            fixed.append(st)
    return ok, notes, fixed
