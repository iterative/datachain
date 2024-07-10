from sqlalchemy.sql.functions import ReturnTypeFromArgs


class greatest(ReturnTypeFromArgs):  # noqa: N801
    inherit_cache = True


class least(ReturnTypeFromArgs):  # noqa: N801
    inherit_cache = True
