import logging
import os


def process_info(record: logging.LogRecord) -> logging.LogRecord:
    """Add process id, threadname and taskname to the log record"""
    if record.process:
        proc = [str(record.process)]
    if record.threadName and record.threadName != "MainThread":
        proc.append(record.threadName)
    if task_name := getattr(record, "taskName", None):
        proc.append(task_name)
    record.process_info = "[" + "/".join(proc) + "]"
    return record


def setup_logging() -> None:
    logger = logging.getLogger("datachain")
    level = logging.WARNING
    if log_level := os.getenv("DATACHAIN_LOG_LEVEL"):
        level = getattr(logging, log_level.upper())
        logger.setLevel(level)

    if "DATACHAIN_LOG_OUTPUT" not in os.environ:
        return

    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s %(process_info)s"
    handler = logging.StreamHandler()
    handler.addFilter(process_info)
    logging.basicConfig(format=format, handlers=[handler], datefmt="%X")

    for module in os.getenv("DATACHAIN_LOG_EXTERNAL", "").split(","):
        module = module.strip()
        if module:
            logging.getLogger(module).setLevel(level)
