# storage rm

Delete files and directories in storage using Studio.

## Synopsis

```usage
usage: datachain storage rm [-h] [-v] [-q] [--recursive] [--team TEAM] path
```

## Description

This command deletes files and directories within storage using the credentials configured in Studio. The command supports both individual files and directories, with the `--recursive` flag required for deleting directories. This is a destructive operation that permanently removes files and cannot be undone.

## Arguments

* `path` - Path to the storage file or directory to delete

## Options

* `--recursive` - Delete directories recursively (required for deleting directories)
* `--team TEAM` - Team name to delete storage contents from (default: from config)
* `-h`, `--help` - Show the help message and exit
* `-v`, `--verbose` - Be verbose
* `-q`, `--quiet` - Be quiet

## Examples

1. Delete a single file:
```bash
datachain storage rm s3://my-bucket/data/file.txt
```

2. Delete a directory recursively:
```bash
datachain storage rm --recursive s3://my-bucket/data/images
```

3. Delete a file from a different team's storage:
```bash
datachain storage rm --team other-team s3://my-bucket/data/file.txt
```

4. Delete a file with verbose output:
```bash
datachain storage rm -v s3://my-bucket/data/file.txt
```

5. Delete a directory quietly (suppress output):
```bash
datachain storage rm -q --recursive s3://my-bucket/temp-data
```

6. Delete a specific subdirectory:
```bash
datachain storage rm --recursive s3://my-bucket/datasets/raw/old-version
```

## Supported Storage Protocols

The command supports the following storage protocols:
- **AWS S3**: `s3://bucket-name/path`
- **Google Cloud Storage**: `gs://bucket-name/path`
- **Azure Blob Storage**: `az://container-name/path`

## Limitations and Edge Cases

### Directory Operations
- **Recursive flag required**: Deleting directories requires the `--recursive` flag. Without it, the operation will fail
- **Directory structure**: When deleting directories, all files and subdirectories within the directory are removed

### File Operations
- **Non-existent files**: Attempting to delete a non-existent file will result in an error
- **Relative vs absolute paths**: Both relative and absolute paths within the bucket are supported

### Error Handling
- **File not found**: If the source file or directory doesn't exist, the operation will fail
- **Permission errors**: Insufficient permissions will result in operation failure
- **Storage service errors**: Network issues or storage service problems will be reported with appropriate error messages
- **Directory not empty**: Attempting to delete a non-empty directory without `--recursive` will fail

### Team Configuration
- **Default team**: If no team is specified, the command uses the team from your configuration
- **Team-specific storage**: Each team has its own storage namespace, so deleting from other teams requires explicit team specification

### Safety Considerations
- **Permanent deletion**: This operation permanently removes files and cannot be undone
- **Batch operations**: Large directories may contain many files and deletion may take time

## Notes

* The delete operation is performed through Studio using the configured credentials
* Deleting large directories may take time depending on the number of files and network conditions
* Use the `--verbose` flag to get detailed information about the delete operation
* The `--quiet` flag suppresses output except for errors
* This command operates through Studio, so you must be authenticated with `datachain auth login` before using it
* **Warning**: This is a destructive operation. Always double-check the path before executing the command
