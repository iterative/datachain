# auth logout

Remove the Studio authentication token from DataChain configuration.

## Synopsis

```usage
usage: datachain auth logout [-h] [-v] [-q] [--local]
```

## Description

This command removes the Studio authentication token from the global DataChain configuration. By default, it removes the token from the global configuration, but you can also remove it from the local project configuration using the `--local` option.

## Options

* `--local` - Remove the token from the local project config instead of the global configuration.
* `-h`, `--help` - Show the help message and exit.
* `-v`, `--verbose` - Be verbose.
* `-q`, `--quiet` - Be quiet.

## Examples

1. Remove token from global configuration:
```bash
datachain auth logout
```

2. Remove token from local project configuration:
```bash
datachain auth logout --local
```

3. Remove token with verbose output:
```bash
datachain auth logout -v
```
