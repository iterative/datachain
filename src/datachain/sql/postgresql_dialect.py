"""
PostgreSQL dialect registration for DataChain.
"""

from datachain.sql.postgresql_types import PostgreSQLTypeConverter
from datachain.sql.types import register_backend_types

# Register PostgreSQL type converter
register_backend_types("postgresql", PostgreSQLTypeConverter())
