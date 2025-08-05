# cp

Copy storage files and directories between cloud and local storage.

## Synopsis

```usage
usage: datachain cp [-h] [-v] [-q] [-r] [--team TEAM]
    [-s] [--anon] [--update] [--no-glob]
    [--force] source_path destination_path
```

## Description

This command copies files and directories between local and/or remote storage. This uses the credentials in your system by default or can use the cloud authentication from Studio.

## Arguments

* `source_path` - Path to the source file or directory to copy
* `destination_path` - Path to the destination file or directory to copy to

## Options

* `-r`, `-R`, `--recursive` - Copy directories recursively
* `--team TEAM` - Team name to use the credentials from.
* `-s`, `--studio-cloud-auth` - Use credentials from Studio for cloud operations (Default: False)
* `--anon` - Use anonymous access to storage
* `--update` - Update cached list of files for the source when downloading from cloud using local credentials.
* `--no-glob` - Do not expand globs (such as * or ?) when downloading from cloud using local credentials.
* `--force` - Force creating files even if they already exist
* `-h`, `--help` - Show the help message and exit
* `-v`, `--verbose` - Be verbose
* `-q`, `--quiet` - Be quiet

## Copy Operations

The command supports two main modes of operation:

### Default Mode
By default, the command operates directly with clouds using credentials in ypur system, supporting various copy scenarios between local and remote storage.

### Studio Cloud Auth Mode
When using `-s` or `--studio-cloud-auth` flag, the command uses credentials from Studio for cloud operations. This mode provides enhanced authentication and access control for cloud storage operations.

## Supported Storage Protocols

The command supports the following storage protocols:
- **Local file system**: Direct paths (e.g., `/path/to/directory` or `./relative/path`)
- **AWS S3**: `s3://bucket-name/path`
- **Google Cloud Storage**: `gs://bucket-name/path`
- **Azure Blob Storage**: `az://container-name/path`

## Examples

The command automatically determines the operation type based on the source and destination protocols:

### Local to Local

**Operation**: Direct local file system copy
- Uses the local filesystem's native copy operation
- Fastest operation as no network transfer is involved
- Supports both files and directories

```bash
datachain cp /path/to/source/file.py /path/to/destination/file.py
datachain cp -r /path/to/source/directory /path/to/destination/directory
```

### Local to Remote

**Operation**: Upload to cloud storage
- Uploads local files/directories to remote storage
- Supports both default mode and Studio cloud auth mode
- Requires `--recursive` flag for directories

```bash
# Upload single file
datachain cp /path/to/local/file.py gs://my-bucket/data/file.py

# Upload single file with Studio cloud auth
datachain cp /path/to/local/file.py gs://my-bucket/data/file.py --studio-cloud-auth

# Upload directory recursively
datachain cp --recursive /path/to/local/directory gs://my-bucket/data/

# Upload directory recursively with Studio cloud auth
datachain cp --recursive /path/to/local/directory gs://my-bucket/data/ --studio-cloud-auth
```

### Remote to Local

**Operation**: Download from cloud storage
- Downloads remote files/directories to local storage
- Automatically extracts filename if destination is a directory
- Creates destination directory if it doesn't exist

```bash
# Download single file
datachain cp gs://my-bucket/data/file.py /path/to/local/directory/

# Download single file with Studio cloud auth
datachain cp gs://my-bucket/data/file.py /path/to/local/directory/ --studio-cloud-auth

# Download directory recursively
datachain cp -r gs://my-bucket/data/directory /path/to/local/directory/
```

### Remote to Remote

**Operation**: Copy within cloud storage
- Copies files between locations in the same bucket
- Cannot copy between different buckets (same limitation as `mv`)
- Requires `--recursive` flag for directories

```bash
# Copy within same bucket
datachain cp gs://my-bucket/data/file.py gs://my-bucket/archive/file.py

# Copy within same bucket with Studio cloud auth
datachain cp gs://my-bucket/data/file.py gs://my-bucket/archive/file.py --studio-cloud-auth
```

### Additional Examples

```bash
# Copy with specific team:
datachain cp -s --team other-team /path/to/file.txt s3://my-bucket/data/file.txt
```


## Limitations
- **Cannot copy between different buckets**: Remote-to-remote copies must be within the same bucket

## Notes
* When using Studio cloud auth mode, you must be authenticated with `datachain auth login` before using it
* The default mode operates directly with storage providers
