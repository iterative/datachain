import ast


class SubclassFinder(ast.NodeVisitor):
    """Finds subclasses of a target class in an AST."""

    def __init__(self, target_classes: list[str]):
        self.imports: list[ast.AST] = []
        self.main_body: list[ast.AST] = []

        self.target_classes: list[str] = target_classes
        self.aliases: dict[str, str] = {}
        self.feature_class: list[ast.AST] = []

    def visit_ImportFrom(self, node):  # noqa: N802
        module = node.module
        for alias in node.names:
            full_name = f"{module}.{alias.name}"
            self.aliases[alias.asname or alias.name] = full_name
        self.imports.append(node)

    def visit_Import(self, node):  # noqa: N802
        for alias in node.names:
            self.aliases[alias.asname or alias.name] = alias.name
        self.imports.append(node)

    def visit_ClassDef(self, node):  # noqa: N802
        base_names = [self.get_base_name(base) for base in node.bases]
        if any(self.is_subclass(name) for name in base_names):
            self.feature_class.append(node)
        else:
            self.main_body.append(node)

    def visit(self, node):
        if isinstance(
            node,
            (ast.Import, ast.ImportFrom, ast.ClassDef, ast.Module),
        ):
            return super().visit(node)
        self.main_body.append(node)
        return node

    def get_base_name(self, node):
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return self.get_full_attr_name(node)
        if isinstance(node, ast.Subscript):
            return self.get_base_name(node.value)
        return None

    def get_full_attr_name(self, node):
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        if isinstance(node.value, ast.Attribute):
            return f"{self.get_full_attr_name(node.value)}.{node.attr}"
        return node.attr

    def is_subclass(self, base_name):
        return base_name and base_name.split(".")[-1] in self.target_classes
