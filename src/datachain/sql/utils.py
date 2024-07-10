from sqlalchemy.ext.compiler import compiles


def compiler_not_implemented(func, *spec):
    package = getattr(func, "package", None)
    if package is None:
        func_identifier = func.name
    else:
        func_identifier = f"{func.package}.{func.name}"

    @compiles(func, *spec)
    def raise_not_implemented(element, compiler, **kwargs):
        try:
            dialect_name = compiler.dialect.name
        except AttributeError:
            dialect_name = "unknown"
        raise NotImplementedError(
            f"Compiler not implemented for the SQLAlchemy function, {func_identifier},"
            f" with dialect, {dialect_name}. For information on adding dialect-specific"
            " compilers, see https://docs.sqlalchemy.org/en/14/core/compiler.html"
        )

    return raise_not_implemented
