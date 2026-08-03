from __future__ import annotations

import ast
import re
from dataclasses import dataclass


@dataclass(slots=True)
class ParsedFile:
    symbols: list[str]
    imports: list[str]
    parser: str


def parse_python(content: str) -> ParsedFile:
    tree = ast.parse(content)
    symbols: list[str] = []
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(node.name)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return ParsedFile(symbols, imports, "python-ast")


def parse_javascript(content: str) -> ParsedFile:
    """Conservative top-level lexical extraction; never advertised as a complete parse."""
    imports = []
    for match in re.finditer(
        r"(?m)^\s*(?:import(?:[^'\"]*from\s*)?|require\s*\()\s*['\"]([^'\"]+)", content
    ):
        imports.append(match.group(1))
    symbols = [
        match.group(1)
        for match in re.finditer(
            r"(?m)^\s*(?:export\s+)?(?:async\s+)?(?:function|class)\s+([A-Za-z_$][\w$]*)", content
        )
    ]
    return ParsedFile(symbols, imports, "javascript-conservative-lexical")


def parse_source(path: str, content: str) -> ParsedFile:
    if path.endswith(".py"):
        try:
            return parse_python(content)
        except SyntaxError:
            return ParsedFile([], [], "python-ast-error")
    if path.endswith((".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")):
        return parse_javascript(content)
    return ParsedFile([], [], "generic")
