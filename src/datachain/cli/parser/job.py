def add_jobs_parser(subparsers, parent_parser) -> None:
    jobs_help = "Manage jobs in Studio"
    jobs_description = "Commands to manage job execution in Studio."
    jobs_parser = subparsers.add_parser(
        "job", parents=[parent_parser], description=jobs_description, help=jobs_help
    )
    jobs_subparser = jobs_parser.add_subparsers(
        dest="cmd",
        help="Use `datachain studio CMD --help` to display command-specific help",
        required=True,
    )

    studio_run_help = "Run a job in Studio"
    studio_run_description = "Run a job in Studio."

    studio_run_parser = jobs_subparser.add_parser(
        "run",
        parents=[parent_parser],
        description=studio_run_description,
        help=studio_run_help,
    )

    studio_run_parser.add_argument(
        "query_file",
        action="store",
        help="Query file to run",
    )

    studio_run_parser.add_argument(
        "--team",
        action="store",
        default=None,
        help="Team to run job for (default: from config)",
    )
    studio_run_parser.add_argument(
        "--env-file",
        action="store",
        help="File with environment variables for the job",
    )

    studio_run_parser.add_argument(
        "--env",
        nargs="+",
        help="Environment variables in KEY=VALUE format",
    )

    studio_run_parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers for the job",
    )
    studio_run_parser.add_argument(
        "--files",
        nargs="+",
        help="Additional files to include in the job",
    )
    studio_run_parser.add_argument(
        "--python-version",
        action="store",
        help="Python version for the job (e.g., 3.9, 3.10, 3.11)",
    )
    studio_run_parser.add_argument(
        "--req-file",
        action="store",
        help="Python requirements file",
    )

    studio_run_parser.add_argument(
        "--req",
        nargs="+",
        help="Python package requirements",
    )

    studio_cancel_help = "Cancel a job in Studio"
    studio_cancel_description = "Cancel a running job in Studio."

    studio_cancel_parser = jobs_subparser.add_parser(
        "cancel",
        parents=[parent_parser],
        description=studio_cancel_description,
        help=studio_cancel_help,
    )

    studio_cancel_parser.add_argument(
        "job_id",
        action="store",
        help="Job ID to cancel",
    )
    studio_cancel_parser.add_argument(
        "--team",
        action="store",
        default=None,
        help="Team to cancel job for (default: from config)",
    )

    studio_log_help = "Show job logs and status in Studio"
    studio_log_description = "Display logs and current status of jobs in Studio."

    studio_log_parser = jobs_subparser.add_parser(
        "logs",
        parents=[parent_parser],
        description=studio_log_description,
        help=studio_log_help,
    )

    studio_log_parser.add_argument(
        "job_id",
        action="store",
        help="Job ID to show logs for",
    )
    studio_log_parser.add_argument(
        "--team",
        action="store",
        default=None,
        help="Team to check logs for (default: from config)",
    )
