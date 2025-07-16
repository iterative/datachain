# cp

Copy storage files and directories between cloud and local storage.

## Synopsis

```usage
usage: datachain cp [-h] [-v] [-q] [-r] [--team TEAM] [--local] [--anon] [--update] [--no-glob] [--force] source_path destination_path
```

## Description

This command copies files and directories between local and/or remote storage. The command can operate through Studio (default) or directly with local storage access.

## Arguments

* `source_path` - Path to the source file or directory to copy
* `destination_path` - Path to the destination file or directory to copy to

## Options

* `-r`, `-R`, `--recursive` - Copy directories recursively
* `--team TEAM` - Team name to copy storage contents to
* `--local` - Copy data files from the cloud locally without Studio (Default: False)
* `--anon` - Use anonymous access to storage (available only with --local)
* `--update` - Update cached list of files for the sources (available only with --local)
* `--no-glob` - Do not expand globs (such as * or ?) (available only with --local)
* `--force` - Force creating files even if they already exist (available only with --local)
* `-h`, `--help` - Show the help message and exit
* `-v`, `--verbose` - Be verbose
* `-q`, `--quiet` - Be quiet

## Copy Operations

The command supports two main modes of operation:

### Studio Mode (Default)
When using Studio mode (default), the command copies files and directories through Studio using the configured credentials. This mode automatically determines the operation type based on the source and destination protocols, supporting four different copy scenarios.

### Local Mode
When using `--local` flag, the command operates directly with local storage access, bypassing Studio. This mode supports additional options like `--anon`, `--update`, `--no-glob`, and `--force`.

## Supported Storage Protocols

The command supports the following storage protocols:
- **Local file system**: Direct paths (e.g., `/path/to/directory` or `./relative/path`)
- **AWS S3**: `s3://bucket-name/path`
- **Google Cloud Storage**: `gs://bucket-name/path`
- **Azure Blob Storage**: `az://container-name/path`

## Examples

### Studio Mode Examples

The command automatically determines the operation type based on the source and destination protocols:

#### 1. Local to Local (local path → local path)
**Operation**: Direct local file system copy
- Uses the local filesystem's native copy operation
- Fastest operation as no network transfer is involved
- Supports both files and directories

```bash
datachain cp /path/to/local/file.txt /path/to/destination/file.txt
```

#### 2. Local to Remote (local path → `s3://`, `gs://`, `az://`)
**Operation**: Upload to cloud storage
- Uploads local files/directories to remote storage
- Uses presigned URLs for secure uploads
- Supports S3 multipart form data for large files
- Requires `--recursive` flag for directories

```bash
# Upload single file
datachain cp /path/to/file.txt s3://my-bucket/data/file.txt

# Upload directory recursively
datachain cp -r /path/to/directory s3://my-bucket/data/
```

#### 3. Remote to Local (`s3://`, `gs://`, `az://` → local path)
**Operation**: Download from cloud storage
- Downloads remote files/directories to local storage
- Uses presigned download URLs
- Automatically extracts filename if destination is a directory
- Creates destination directory if it doesn't exist

```bash
# Download single file
datachain cp s3://my-bucket/data/file.txt /path/to/local/file.txt

# Download to directory (filename preserved)
datachain cp s3://my-bucket/data/file.txt /path/to/directory/
```

#### 4. Remote to Remote (`s3://` → `s3://`, `gs://` → `gs://`, etc.)
**Operation**: Copy within cloud storage
- Copies files between locations in the same bucket
- Cannot copy between different buckets (same limitation as `mv`)
- Uses Studio's internal copy operation
- Requires `--recursive` flag for directories

```bash
# Copy within same bucket
datachain cp s3://my-bucket/data/file.txt s3://my-bucket/archive/file.txt

# Copy directory recursively
datachain cp -r s3://my-bucket/data/images s3://my-bucket/backup/images
```

### Additional Studio Mode Examples

1. Copy with specific team:
```bash
datachain cp --team other-team /path/to/file.txt s3://my-bucket/data/file.txt
```

2. Copy with verbose output:
```bash
datachain cp -v -r s3://my-bucket/datasets/raw s3://my-bucket/datasets/processed
```

### Local Mode Examples

3. Copy files locally without Studio:
```bash
datachain cp --local /path/to/source /path/to/destination
```

4. Copy with anonymous access:
```bash
datachain cp --local --anon s3://public-bucket/data /path/to/local/
```

5. Copy with force overwrite:
```bash
datachain cp --local --force s3://my-bucket/data /path/to/local/
```

6. Copy with update and no glob expansion:
```bash
datachain cp --local --update --no-glob s3://my-bucket/data/*.txt /path/to/local/
```

## Limitations and Edge Cases

### Bucket Restrictions
- **Cannot copy between different buckets**: Remote-to-remote copies must be within the same bucket
- **Cross-bucket operations**: Use local as intermediate step for cross-bucket copies

### Directory Operations
- **Recursive flag required**: Copying directories requires the `--recursive` flag
- **Directory structure preservation**: Directory structure is preserved during copy operations
- **Empty directories**: Empty directories may not be copied in some scenarios


### Error Handling
- **File not found**: Missing source files result in operation failure
- **Permission errors**: Insufficient permissions cause operation failure
- **Network issues**: Network problems are reported with appropriate error messages

### Team Configuration
- **Default team**: If no team is specified, uses the team from your configuration
- **Team-specific storage**: Each team has its own storage namespace

## Notes

* Use the `--verbose` flag to get detailed information about the copy operation
* The `--quiet` flag suppresses output except for errors
* When using Studio mode, you must be authenticated with `datachain auth login` before using it
* The `--local` mode bypasses Studio and operates directly with storage providers
* Use `--recursive` flag when copying directories
* The `--force` flag is only available in local mode and will overwrite existing files
* For cross-bucket copies, consider using local storage as an intermediate step
