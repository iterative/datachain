# Writing DataChain UDFs

UDFs are created by applying the `@udf` decorator to functions or classes:

``` python
@udf(
    params=("name",),              # Columns consumed by the UDF.
    output={"path_len": Integer},  # Signals being returned by the UDF, with the signal name and type.
)
def name_len(name):
    return (len(name),)
```

The decorator takes several parameters:
 - `params` - a sequence which will be passed to the UDF as parameters
 - `output` - a dictionary containing a signal name and the sqlalchemy type for the signal
 - `method` - an optional parameter specifying the method of a class UDF to call
 - `batch` - an optional number of batches of inputs to process with each UDF call

## Specifying parameters

UDF parameters refer to the columns in an index or a dataset or the actual file object in storage.
Columns are specified as `"name"`, `C.name`, `"parent"`, `C.parent`, `etc.`.

Objects are specified as `Object(loader)` where the `loader` is a function to load the object file.
For example to load a file as a PIL image, the loader can be defined as:

``` python
def load_image(raw):
    img = Image.open(raw)
    img.load()
    return img
```

and then used as `Object(load_image)`.

## UDF return values

UDFs need to bundle their return values in tuples (or lists of tuples for batched UDFs).
So a UDF returning a single value should return `(value,)`, for multiple value returns -
`(value1, value2)`.

## Batching

In some instances it makes sense to call UDFs for sets of rows, not once for each row.
This is done by specifying the `batch=n` parameter in the `@udf` decorator. In this case
inputs are bundled in sequences of no more than `n` tuples containing values of the input
parameters.

Batching may be useful in stateful UDFs (see below) to load models or other resources
in `__init__` method and then use them for multiple rows.

``` python
@udf(
    output=(("path_len", Integer),),
    parameters=(
        C.parent,
        C.name,
    ),
    batch=10,
)
def name_len(names):
    return [(len(parent + name),) for (parent, name) in names]
```

Batched UDF functions take a single parameter.

## Stateful (class) UDFs

In some cases UDFs require instantiation (e.g. to load models). In such cases it makes
sense to write the UDF as a class and decorate it with the same `@udf` decorator.

``` python
@udf(
    output=(("path_len", Integer),),
    parameters=(
        C.parent,
        C.name,
    ),
    method="name_len",
)
class NameLen:
    def __init__(self, multiplier=1):
        self.multiplier = multiplier

    def name_len(self, parent, name):
        return (len(parent + name)*self.multiplier,)
```

The UDF can then be passed to the `add_signals` call:

``` python
DatasetQuery(name).add_signals(NameLen(multiplier=2))
```
