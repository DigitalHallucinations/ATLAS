"""Structured documentation helpers for DocGenius."""

from __future__ import annotations

import ast
import asyncio
from typing import Dict, Optional


def _annotation_to_str(node: Optional[ast.expr]) -> str:
    if node is None:
        return "unspecified"
    try:
        return ast.unparse(node)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - fallback path for Python <3.9 behaviour
        if isinstance(node, ast.Attribute):
            value = _annotation_to_str(node.value)
            return f"{value}.{node.attr}" if value else node.attr
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Subscript):
            return f"{_annotation_to_str(node.value)}[{_annotation_to_str(node.slice)}]"
        return "unspecified"


def _parse_python_signature(signature: str) -> Dict[str, object]:
    candidate = signature.strip()
    if candidate and not candidate.endswith(":"):
        candidate = f"{candidate}:"
    if candidate.endswith(":"):
        candidate = f"{candidate}\n    ..."

    tree = ast.parse(candidate)
    node = next(
        (
            child
            for child in tree.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ),
        None,
    )

    if node is None:
        raise ValueError("Unable to locate a definition in the provided signature.")

    if isinstance(node, ast.ClassDef):
        params = [
            {
                "name": base.id if isinstance(base, ast.Name) else getattr(base, "attr", "object"),
                "type": "base class",
                "description": "Explain how this parent contributes responsibilities.",
            }
            for base in node.bases
        ]
        returns: Optional[str] = None
        entity_name = node.name
        entity_kind = "class"
    else:
        params = [
            {
                "name": arg.arg,
                "type": _annotation_to_str(arg.annotation),
                "description": "Describe this parameter's intent and constraints.",
            }
            for arg in [*node.args.posonlyargs, *node.args.args]
        ]
        if node.args.vararg is not None:
            params.append(
                {
                    "name": f"*{node.args.vararg.arg}",
                    "type": _annotation_to_str(node.args.vararg.annotation),
                    "description": "Document flexible positional arguments.",
                }
            )
        if node.args.kwarg is not None:
            params.append(
                {
                    "name": f"**{node.args.kwarg.arg}",
                    "type": _annotation_to_str(node.args.kwarg.annotation),
                    "description": "Clarify accepted keyword expansions.",
                }
            )
        params.extend(
            {
                "name": kw.arg or "keyword_only",
                "type": _annotation_to_str(kw.annotation),
                "description": "Detail this keyword-only argument.",
            }
            for kw in node.args.kwonlyargs
        )
        returns = _annotation_to_str(node.returns)
        entity_name = node.name
        entity_kind = "function"

    return {
        "entity": {
            "name": entity_name,
            "kind": entity_kind,
        },
        "parameters": params,
        "returns": returns,
    }


async def generate_doc_outline(signature: str, language: str = "python") -> Dict[str, object]:
    """Generate the DocGenius documentation scaffold for a symbol definition.

    Parameters
    ----------
    signature:
        Raw definition snippet (function or class) that should be documented.
    language:
        Source language for ``signature``. Python parsing is supported; other
        values fall back to a lightweight heuristic outline.

    Returns
    -------
    Dict[str, object]
        Structured payload containing summary guidance, parameter prompts, and
        follow-up hooks that match DocGenius' operator playbooks.

    Examples
    --------
    >>> await generate_doc_outline('def add(a: int, b: int) -> int:')
    {'entity': {'name': 'add', 'kind': 'function'}, ...}  # doctest: +SKIP
    """

    await asyncio.sleep(0)
    language_key = language.lower()

    if language_key == "python":
        parsed = _parse_python_signature(signature)
    else:
        parsed = {
            "entity": {
                "name": signature.strip().split()[0],
                "kind": "symbol",
            },
            "parameters": [],
            "returns": None,
        }

    sections: Dict[str, object] = {
        "Summary": "Articulate the high-level intent and context for this symbol.",
        "Parameters": parsed["parameters"],
        "Returns": parsed["returns"],
        "Raises": [
            {
                "exception": "",
                "description": "List expected failure modes and user-facing impact.",
            }
        ],
        "Examples": [
            "Provide a minimal, copy-paste friendly example aligned with production usage.",
        ],
        "FollowUp": [
            "Note documentation TODOs or additional references to curate.",
        ],
    }

    return {
        **parsed,
        "language": language_key,
        "sections": sections,
        "signature": signature.strip(),
    }
