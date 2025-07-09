# storage cp

Copy files and directories between local and/or remote storage using Studio.

## Synopsis

```usage
usage: datachain storage cp [-h] [-v] [-q] [--recursive] [--team TEAM] source_path destination_path
```

## Description

This command copies files and directories between local and/or remote storage using the credentials configured in Studio. The command automatically determines the operation type based on the source and destination protocols, supporting four different copy scenarios.

## Arguments

* `source_path` - Path to the source file or directory to copy
* `destination_path` - Path to the destination file or directory to copy to

## Options

* `--recursive` - Copy directories recursively (required for copying directories)
* `--team TEAM` - Team name to copy storage contents to (default: from config)
* `-h`, `--help` - Show the help message and exit
* `-v`, `--verbose` - Be verbose
* `-q`, `--quiet` - Be quiet

## Copy Operations

The command automatically determines the operation type based on the source and destination protocols:

### 1. Local to Local (local path → local path)
**Operation**: Direct local file system copy
- Uses the local filesystem's native copy operation
- Fastest operation as no network transfer is involved
- Supports both files and directories

**Example**:
```bash
datachain storage cp /path/to/local/file.txt /path/to/destination/file.txt
```

### 2. Local to Remote (local path → `s3://`, `gs://`, `az://`)
**Operation**: Upload to cloud storage
- Uploads local files/directories to remote storage
- Uses presigned URLs for secure uploads
- Supports S3 multipart form data for large files
- Requires `--recursive` flag for directories

**Examples**:
```bash
# Upload single file
datachain storage cp /path/to/file.txt s3://my-bucket/data/file.txt

# Upload directory recursively
datachain storage cp --recursive /path/to/directory s3://my-bucket/data/
```

### 3. Remote to Local (`s3://`, `gs://`, `az://` → local path)
**Operation**: Download from cloud storage
- Downloads remote files/directories to local storage
- Uses presigned download URLs
- Automatically extracts filename if destination is a directory
- Creates destination directory if it doesn't exist

**Examples**:
```bash
# Download single file
datachain storage cp s3://my-bucket/data/file.txt /path/to/local/file.txt

# Download to directory (filename preserved)
datachain storage cp s3://my-bucket/data/file.txt /path/to/directory/
```

### 4. Remote to Remote (`s3://` → `s3://`, `gs://` → `gs://`, etc.)
**Operation**: Copy within cloud storage
- Copies files between locations in the same bucket
- Cannot copy between different buckets (same limitation as `mv`)
- Uses Studio's internal copy operation
- Requires `--recursive` flag for directories

**Examples**:
```bash
# Copy within same bucket
datachain storage cp s3://my-bucket/data/file.txt s3://my-bucket/archive/file.txt

# Copy directory recursively
datachain storage cp --recursive s3://my-bucket/data/images s3://my-bucket/backup/images
```

## Supported Storage Protocols

The command supports the following storage protocols:
- **Local file system**: Direct paths (e.g., `/path/to/directory` or `./relative/path`)
- **AWS S3**: `s3://bucket-name/path`
- **Google Cloud Storage**: `gs://bucket-name/path`
- **Azure Blob Storage**: `az://container-name/path`

## Examples

### Local to Remote Operations

1. Upload a single file:
```bash
datachain storage cp /path/to/image.jpg s3://my-bucket/images/image.jpg
```

2. Upload a directory recursively:
```bash
datachain storage cp --recursive /path/to/dataset s3://my-bucket/datasets/
```

3. Upload to a different team's storage:
```bash
datachain storage cp --team other-team /path/to/file.txt s3://my-bucket/data/file.txt
```

### Remote to Local Operations

4. Download a file:
```bash
datachain storage cp s3://my-bucket/data/file.txt /path/to/local/file.txt
```

### Remote to Remote Operations

6. Copy within the same bucket:
```bash
datachain storage cp s3://my-bucket/data/file.txt s3://my-bucket/archive/file.txt
```

7. Copy directory with verbose output:
```bash
datachain storage cp -v --recursive s3://my-bucket/datasets/raw s3://my-bucket/datasets/processed
```

## Limitations and Edge Cases

### Bucket Restrictions
- **Cannot copy between different buckets**: Remote-to-remote copies must be within the same bucket
- **Cross-bucket operations**: Use local as intermediate step for cross-bucket copies

### Directory Operations
- **Recursive flag required**: Copying directories requires the `--recursive` flag
- **Directory structure preservation**: Directory structure is preserved during copy operations
- **Empty directories**: Empty directories may not be copied in some scenarios

### File Operations
- **File overwrites**: Existing files may be overwritten without confirmation
- **Relative vs absolute paths**: Both relative and absolute paths are supported
- **Directory creation**: Destination directories are created automatically when needed

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
* This command operates through Studio, so you must be authenticated with `datachain auth login` before using it
* For cross-bucket copies, consider using local storage as an intermediate step
