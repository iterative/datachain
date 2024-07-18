# UDF

User-defined functions (UDFs) can run batch processing on a chain to generate new chain
values. A UDF can be any Python function. The UDF will take fields from one or more rows
of the data and output new fields. A UDF can run at scale on multiple workers and
processes.

::: datachain.lib.udf.Aggregator

::: datachain.lib.udf.Generator

::: datachain.lib.udf.Mapper
