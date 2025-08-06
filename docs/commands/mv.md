# mv

Move storage files and directories in clouds or local filesystem.

## Synopsis

```usage
usage: datachain mv [-h] [-v] [-q] [--recursive]
     [--team TEAM] [-s] path new_path
```

## Description

This command moves files and directories within storage. The command supports both individual files and directories, with the `--recursive` flag required for moving directories.

## Arguments

* `path` - Path to the storage file or directory to move
* `new_path` - New path where the file or directory should be moved to

## Options

* `--recursive` - Move recursively
* `--team TEAM` - Team name to use the credentials from. (Default: from config)
* `-s`, `--studio-cloud-auth` - Use credentials from Studio for cloud operations (Default: False)
* `-h`, `--help` - Show the help message and exit
* `-v`, `--verbose` - Be verbose
* `-q`, `--quiet` - Be quiet

## Examples

The command supports moving files and directories within the same bucket:

## Notes
* When using Studio cloud auth mode, you must be authenticated with `datachain auth login` before using it
* The default mode operates directly with storage providers
* **Warning**: This is a destructive operation. Always double-check the path before executing the command

### Move Single File

```bash
# Move file
datachain mv gs://my-bucket/data/file.py gs://my-bucket/archive/file.py

# Move file with Studio cloud auth
datachain mv gs://my-bucket/data/file.py gs://my-bucket/archive/file.py --studio-cloud-auth
```

### Move Directory Recursively

```bash
# Move directory
datachain mv gs://my-bucket/data/directory gs://my-bucket/archive/directory --recursive

# Move directory with Studio cloud auth
datachain mv gs://my-bucket/data/directory gs://my-bucket/archive/directory --recursive --studio-cloud-auth
```

### Additional Examples

```bash
# Move a file to a different team's storage:
datachain mv -s --team other-team gs://my-bucket/data/file.py gs://my-bucket/backup/file.py
```


## Supported Storage Protocols

The command supports the following storage protocols:
- **AWS S3**: `s3://bucket-name/path`
- **Google Cloud Storage**: `gs://bucket-name/path`
- **Azure Blob Storage**: `az://container-name/path`

## Limitations and Edge Cases
- **Cannot move between different buckets**: The source and destination must be in the same bucket. Attempting to move between different buckets will result in an error: "Cannot move between different buckets"

## Notes
* When using Studio cloud auth mode, you must be authenticated with `datachain auth login` before using it
* The default mode operates directly with storage providers
