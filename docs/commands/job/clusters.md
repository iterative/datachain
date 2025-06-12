# job clusters

List compute clusters in Studio.

## Synopsis

```usage
usage: datachain job clusters [-h] [-v] [-q] [--team TEAM]
```

## Description

This command lists compute clusters available in Studio. You can specify a team to list clusters for. The command provides information about the compute resources available for running jobs.

## Options

* `--team TEAM` - Team to list clusters for (default: from config)
* `-h`, `--help` - Show the help message and exit
* `-v`, `--verbose` - Be verbose
* `-q`, `--quiet` - Be quiet

## Examples

1. List all clusters for the default team:
```bash
datachain job clusters
```

2. List clusters for a specific team:
```bash
datachain job clusters --team my-team
```


## Notes

* The command shows all compute clusters available to your team
* Clusters represent the compute resources where your jobs can run
* Use the `--team` option to view clusters for a different team
