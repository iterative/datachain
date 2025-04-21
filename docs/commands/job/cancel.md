# job cancel

Cancel a running job in Studio.

## Synopsis

```usage
usage: datachain job cancel [-h] [-v] [-q] [--team TEAM] id
```

## Description

This command cancels a running job in Studio. The job ID can be obtained from the Studio UI or from the output when the job was created. This is the recommended way to stop a running job, as simply closing the logs view (e.g., with Ctrl+C) will not cancel the job execution.

## Arguments

* `id` - Job ID to cancel. This ID is displayed when the job is created and can also be found in the Studio UI.

## Options

* `--team TEAM` - Team to cancel job for (default: from config)
* `-h`, `--help` - Show the help message and exit.
* `-v`, `--verbose` - Be verbose.
* `-q`, `--quiet` - Be quiet.

## Examples

1. Cancel a specific job:
```bash
datachain job cancel job-123
```

2. Cancel a job in a specific team:
```bash
datachain job cancel --team my-team job-123
```


## Notes

* The job ID is displayed when the job is created using `datachain job run`
* You can also find the job ID in the Studio UI
* This is the proper way to stop a running job, as simply closing the logs view will not cancel the job execution
