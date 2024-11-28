from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql import expression, selectable


class Values(selectable.Values):
    def __init__(self, data, columns=None, **kwargs):
        if columns is None:
            num_columns = len(data[0])
            columns = [expression.column(f"c{i}") for i in range(1, num_columns + 1)]
        else:
            columns = [
                process_column_expression(c)
                for c in columns
                # expression.column(c) if isinstance(c, str) else c for c in columns
            ]
        super().__init__(*columns, **kwargs)
        self._data += tuple(data)


def values(data, columns=None, **kwargs) -> Values:
    return Values(data, columns=columns, **kwargs)


def process_column_expression(col):
    if hasattr(col, "get_column"):
        return col.get_column()
    if isinstance(col, str):
        return expression.column(col)
    return col


def select(*columns, **kwargs) -> "expression.Select":
    columns_processed = [process_column_expression(c) for c in columns]
    return expression.select(*columns_processed, **kwargs)


def base_values_compiler(column_name_func, element, compiler, **kwargs):
    columns = element.columns
    base_values = expression.values(*columns).data(element._data)
    col_expressions = [
        expression.column(column_name_func(i)).label(c.name)
        for i, c in enumerate(columns, 1)
    ]
    expr = (
        expression.select(*col_expressions)
        .select_from(base_values)
        .subquery(element.name)
    )
    return compiler.process(expr, **kwargs)


def compile_values(element, compiler, **kwargs):
    return base_values_compiler(lambda i: f"c{i}", element, compiler, **kwargs)


compiles(Values, "default")(compile_values)
