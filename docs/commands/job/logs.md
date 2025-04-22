# job logs

Display logs and current status of jobs in Studio.

## Synopsis

```usage
usage: datachain job logs [-h] [-v] [-q] [--team TEAM] id
```

## Description

This command displays the logs and current status of a running job in Studio. The command will show real-time logs from the job execution. Note that closing this command (e.g., with Ctrl+C) will only stop displaying the logs but will not cancel the job execution. To cancel a job, use the `job cancel` command.

## Arguments

* `id` - Job ID to show logs for

## Options

* `--team TEAM` - Team to check logs for (default: from config)
* `-h`, `--help` - Show the help message and exit.
* `-v`, `--verbose` - Be verbose.
* `-q`, `--quiet` - Be quiet.

## Examples

1. Display logs for a specific job:
```bash
datachain job logs job-123
```

2. Display logs for a job in a specific team:
```bash
datachain job logs --team my-team job-123
```

3. Display logs with verbose output:
```bash
datachain job logs -v job-123
```

## Notes

* Closing the logs command (e.g., with Ctrl+C) will only stop displaying the logs but will not cancel the job execution
* To cancel a running job, use the `datachain job cancel` command
* The job will continue running in Studio even after you stop viewing the logs
