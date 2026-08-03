from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

import tree_sitter_javascript
import tree_sitter_typescript
from tree_sitter import Language, Parser

PARSER_VERSION = "symbols-v2-tree-sitter-0.25"


@dataclass(slots=True)
class SymbolRange:
    name: str
    kind: str
    start_line: int
    end_line: int


@dataclass(slots=True)
class ParsedFile:
    symbols: list[str]
    imports: list[str]
    parser: str
    ranges: list[SymbolRange]


def parse_python(content: str) -> ParsedFile:
    tree = ast.parse(content)
    symbols: list[str] = []
    imports: list[str] = []
    ranges: list[SymbolRange] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(node.name)
            ranges.append(
                SymbolRange(
                    node.name, type(node).__name__, node.lineno, node.end_lineno or node.lineno
                )
            )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    symbols.append(target.id)
                    ranges.append(
                        SymbolRange(
                            target.id, "assignment", node.lineno, node.end_lineno or node.lineno
                        )
                    )
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return ParsedFile(symbols, imports, "python-ast-v2", ranges)


def parse_javascript(content: str, language_name: str = "javascript") -> ParsedFile:
    capsule = (
        tree_sitter_typescript.language_typescript()
        if language_name == "typescript"
        else tree_sitter_javascript.language()
    )
    parser = Parser(Language(capsule))
    raw = content.encode()
    root = parser.parse(raw).root_node
    if root.has_error:
        return ParsedFile([], [], f"tree-sitter-{language_name}-error", [])
    symbols: list[str] = []
    imports: list[str] = []
    ranges: list[SymbolRange] = []

    def text(node: Any) -> str:
        return raw[node.start_byte : node.end_byte].decode()

    def visit(node: Any, top_level: bool = True) -> None:
        if node.type == "import_statement":
            source = node.child_by_field_name("source")
            if source is not None:
                imports.append(text(source).strip("'\""))
        if node.type in {"function_declaration", "class_declaration"}:
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                name = text(name_node)
                symbols.append(name)
                ranges.append(
                    SymbolRange(name, node.type, node.start_point.row + 1, node.end_point.row + 1)
                )
        elif node.type == "variable_declarator" and top_level:
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                name = text(name_node)
                symbols.append(name)
                ranges.append(
                    SymbolRange(
                        name, "declaration", node.start_point.row + 1, node.end_point.row + 1
                    )
                )
        child_top = top_level and node.type in {
            "program",
            "export_statement",
            "lexical_declaration",
            "variable_declaration",
        }
        for child in node.children:
            visit(child, child_top)

    visit(root)
    return ParsedFile(symbols, imports, f"tree-sitter-{language_name}-0.25", ranges)


def parse_source(path: str, content: str) -> ParsedFile:
    if path.endswith(".py"):
        try:
            return parse_python(content)
        except SyntaxError:
            return ParsedFile([], [], "python-ast-error", [])
    if path.endswith((".js", ".jsx", ".mjs", ".cjs")):
        return parse_javascript(content)
    if path.endswith((".ts", ".tsx")):
        return parse_javascript(content, "typescript")
    return ParsedFile([], [], "generic", [])
