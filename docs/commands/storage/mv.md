# storage mv

Move files and directories in Storages using Studio.

## Synopsis

```usage
usage: datachain storage mv [-h] [-v] [-q] [--recursive] [--team TEAM] path new_path
```

## Description

This command moves files and directories within storage using the credentials configured in Studio.. The move operation is performed within the same bucket - you cannot move files between different buckets. The command supports both individual files and directories, with the `--recursive` flag required for moving directories.

## Arguments

* `path` - Path to the storage file or directory to move
* `new_path` - New path where the file or directory should be moved to

## Options

* `--recursive` - Move directories recursively (required for moving directories)
* `--team TEAM` - Team name to move storage contents from (default: from config)
* `-h`, `--help` - Show the help message and exit
* `-v`, `--verbose` - Be verbose
* `-q`, `--quiet` - Be quiet

## Examples

1. Move a single file:
```bash
datachain storage mv s3://my-bucket/data/file.txt s3://my-bucket/archive/file.txt
```

2. Move a directory recursively:
```bash
datachain storage mv --recursive s3://my-bucket/data/images s3://my-bucket/archive/images
```

3. Move a file to a different team's storage:
```bash
datachain storage mv --team other-team s3://my-bucket/data/file.txt s3://my-bucket/backup/file.txt
```

4. Move a file with verbose output:
```bash
datachain storage mv -v s3://my-bucket/data/file.txt s3://my-bucket/processed/file.txt
```

5. Move a directory to a subdirectory:
```bash
datachain storage mv --recursive s3://my-bucket/datasets/raw s3://my-bucket/datasets/processed/raw
```

## Supported Storage Protocols

The command supports the following storage protocols:
- **AWS S3**: `s3://bucket-name/path`
- **Google Cloud Storage**: `gs://bucket-name/path`
- **Azure Blob Storage**: `az://container-name/path`
- **Local file system**: `file:///path/to/directory`

## Limitations and Edge Cases

### Bucket Restrictions
- **Cannot move between different buckets**: The source and destination must be in the same bucket. Attempting to move between different buckets will result in an error: "Cannot move between different buckets"

### Directory Operations
- **Recursive flag required**: Moving directories requires the `--recursive` flag. Without it, the operation will fail
- **Directory structure preservation**: When moving directories, the internal structure is preserved

### Path Handling
- **Relative vs absolute paths**: Both relative and absolute paths within the bucket are supported

### Error Handling
- **File not found**: If the source file or directory doesn't exist, the operation will fail
- **Permission errors**: Insufficient permissions will result in operation failure
- **Storage service errors**: Network issues or storage service problems will be reported with appropriate error messages

### Team Configuration
- **Default team**: If no team is specified, the command uses the team from your configuration
- **Team-specific storage**: Each team has its own storage namespace, so moving between teams is not supported

## Notes

* Moving large directories may take time depending on the number of files and network conditions
* Use the `--verbose` flag to get detailed information about the move operation
* The `--quiet` flag suppresses output except for errors
* This command operates through Studio, so you must be authenticated with `datachain auth login` before using it
