# job run

Run a job in Studio.

## Synopsis

```usage
usage: datachain job run [-h] [-v] [-q] [--team TEAM] [--env-file ENV_FILE] [--env ENV [ENV ...]]
                         [--workers WORKERS] [--files FILES [FILES ...]] [--python-version PYTHON_VERSION]
                         [--req-file REQ_FILE] [--req REQ [REQ ...]]
                         file
```

## Description

This command runs a job in Studio using the specified query file. You can configure various aspects of the job including environment variables, Python version, dependencies, and more.

## Arguments

* `file` - Query file to run

## Options

* `--team TEAM` - Team to run job for (default: from config)
* `--env-file ENV_FILE` - File with environment variables for the job
* `--env ENV` - Environment variables in KEY=VALUE format
* `--workers WORKERS` - Number of workers for the job
* `--files FILES` - Additional files to include in the job
* `--python-version PYTHON_VERSION` - Python version for the job (e.g., 3.9, 3.10, 3.11)
* `--req-file REQ_FILE` - Python requirements file
* `--req REQ` - Python package requirements
* `--priority PRIORITY` - Priority for the job in range 0-5. Lower value is higher priority (default: 5)
* `-h`, `--help` - Show the help message and exit.
* `-v`, `--verbose` - Be verbose.
* `-q`, `--quiet` - Be quiet.

## Examples

1. Run a basic job:
```bash
datachain job run query.py
```

2. Run a job with specific team and Python version:
```bash
datachain job run --team my-team --python-version 3.11 query.py
```

3. Run a job with environment variables and requirements:
```bash
datachain job run --env-file .env --req-file requirements.txt query.py
```

4. Run a job with multiple workers and additional files:
```bash
datachain job run --workers 4 --files utils.py config.json query.py
```

5. Run a job with inline environment variables and package requirements:
```bash
datachain job run --env API_KEY=123 --req pandas numpy query.py
```

6. Run a job with a repository (will be cloned in the job working directory):
```bash
datachain job run --repository https://github.com/iterative/datachain query.py
```

7. Run a job with higher priority
```bash
datachain job run --priority 2 query.py
```

## Notes

* Closing the logs command (e.g., with Ctrl+C) will only stop displaying the logs but will not cancel the job execution
* To cancel a running job, use the `datachain job cancel` command
* The job will continue running in Studio even after you stop viewing the logs
