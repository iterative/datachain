# auth team

Set the default team for Studio operations.

## Synopsis

```usage
usage: datachain auth team [-h] [-v] [-q] [--global] team_name
```

## Description

This command sets the default team for Studio operations. By default, the team setting is project-specific, but you can use the `--global` option to set it for all projects.

## Arguments

* `team_name` - Name of the team to set as default

## Options

* `--global` - Set team globally for all projects
* `-h`, `--help` - Show the help message and exit.
* `-v`, `--verbose` - Be verbose.
* `-q`, `--quiet` - Be quiet.

## Examples

1. Set default team for current project:
```bash
datachain auth team my-team
```

2. Set default team globally for all projects:
```bash
datachain auth team --global my-team
```
