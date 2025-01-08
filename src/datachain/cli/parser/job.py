def add_jobs_parser(subparsers, parent_parser) -> None:
    jobs_help = "Commands to handle the Job running with Iterative Studio"
    jobs_description = (
        "This will help us to run, cancel and view the status of the job in Studio. "
    )
    jobs_parser = subparsers.add_parser(
        "job", parents=[parent_parser], description=jobs_description, help=jobs_help
    )
    jobs_subparser = jobs_parser.add_subparsers(
        dest="cmd",
        help="Use `DataChain studio CMD --help` to display command-specific help.",
        required=True,
    )

    studio_run_help = "Run a job in Studio"
    studio_run_description = "This command runs a job in Studio."

    studio_run_parser = jobs_subparser.add_parser(
        "run",
        parents=[parent_parser],
        description=studio_run_description,
        help=studio_run_help,
    )

    studio_run_parser.add_argument(
        "query_file",
        action="store",
        help="The query file to run.",
    )

    studio_run_parser.add_argument(
        "--team",
        action="store",
        default=None,
        help="The team to run a job for. By default, it will use team from config.",
    )
    studio_run_parser.add_argument(
        "--env-file",
        action="store",
        help="File containing environment variables to set for the job.",
    )

    studio_run_parser.add_argument(
        "--env",
        nargs="+",
        help="Environment variable. Can be specified multiple times. Format: KEY=VALUE",
    )

    studio_run_parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers to use for the job.",
    )
    studio_run_parser.add_argument(
        "--files",
        nargs="+",
        help="Files to include in the job.",
    )
    studio_run_parser.add_argument(
        "--python-version",
        action="store",
        help="Python version to use for the job (e.g. '3.9', '3.10', '3.11').",
    )
    studio_run_parser.add_argument(
        "--req-file",
        action="store",
        help="File containing Python package requirements.",
    )

    studio_run_parser.add_argument(
        "--req",
        nargs="+",
        help="Python package requirement. Can be specified multiple times.",
    )

    studio_cancel_help = "Cancel a job in Studio"
    studio_cancel_description = "This command cancels a job in Studio."

    studio_cancel_parser = jobs_subparser.add_parser(
        "cancel",
        parents=[parent_parser],
        description=studio_cancel_description,
        help=studio_cancel_help,
    )

    studio_cancel_parser.add_argument(
        "job_id",
        action="store",
        help="The job ID to cancel.",
    )
    studio_cancel_parser.add_argument(
        "--team",
        action="store",
        default=None,
        help="The team to cancel a job for. By default, it will use team from config.",
    )

    studio_log_help = "Show the logs and latest status of Jobs in Studio"
    studio_log_description = (
        "This will display the logs and latest status of jobs in Studio"
    )

    studio_log_parser = jobs_subparser.add_parser(
        "logs",
        parents=[parent_parser],
        description=studio_log_description,
        help=studio_log_help,
    )

    studio_log_parser.add_argument(
        "job_id",
        action="store",
        help="The job ID to show the logs.",
    )
    studio_log_parser.add_argument(
        "--team",
        action="store",
        default=None,
        help="The team to check the logs. By default, it will use team from config.",
    )
