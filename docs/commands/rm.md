# rm

Delete storage files and directories from cloud or local system.

## Synopsis

```usage
usage: datachain rm [-h] [-v] [-q] [--recursive] [--team TEAM] [-s] path
```

## Description

This command deletes files and directories within storage. The command supports both individual files and directories, with the `--recursive` flag required for deleting directories. This is a destructive operation that permanently removes files and cannot be undone.

## Arguments

* `path` - Path to the storage file or directory to delete

## Options

* `--recursive` - Delete recursively
* `--team TEAM` - Team name to use the credentials from. (Default: from config)
* `-s`, `--studio-cloud-auth` - Use credentials from Studio for cloud operations (Default: False)
* `-h`, `--help` - Show the help message and exit
* `-v`, `--verbose` - Be verbose
* `-q`, `--quiet` - Be quiet


## Notes
* When using Studio cloud auth mode, you must be authenticated with `datachain auth login` before using it
* The default mode operates directly with storage providers
* **Warning**: This is a destructive operation. Always double-check the path before executing the command


## Examples

The command supports deleting files and directories:

### Delete Single File

```bash
# Delete file
datachain rm gs://my-bucket/data/file.py --recursive

# Delete file with Studio cloud auth
datachain rm gs://my-bucket/data/file.py --studio-cloud-auth
```

### Delete Directory Recursively

```bash
# Delete directory
datachain rm gs://my-bucket/data/directory --recursive

# Delete directory with Studio cloud auth
datachain rm gs://my-bucket/data/directory --recursive --studio-cloud-auth
```
