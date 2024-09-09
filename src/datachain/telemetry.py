import logging
from functools import wraps
from importlib.metadata import PackageNotFoundError, version

from iterative_telemetry import IterativeTelemetryLogger

from datachain.config import read_config
from datachain.utils import DataChainDir, env2bool

logger = logging.getLogger(__name__)


def is_enabled():
    """
    Determine if telemetry is enabled based on environment variables and configuration.
    """
    # Disable telemetry if running in test mode
    if env2bool("DATACHAIN_TEST"):
        return False

    # Check if telemetry is disabled by environment variable
    disabled = env2bool("DATACHAIN_NO_ANALYTICS")
    if disabled:
        logger.debug("Telemetry is disabled by environment variable.")
        return False

    # Check if telemetry is disabled by configuration file
    config = read_config(DataChainDir.find().root)
    if config and config.get("core", {}).get("no_analytics", False):
        logger.debug("Telemetry is disabled by configuration.")
        return False

    logger.debug("Telemetry is enabled.")
    return True


# Try to get the version of the datachain package
try:
    __version__ = version("datachain")
except PackageNotFoundError:
    __version__ = "unknown"

# Initialize telemetry logger
telemetry = IterativeTelemetryLogger("datachain", __version__, is_enabled)
_is_api_running = False
_pass_params = True


def api_telemetry(f):
    """
    Decorator to add telemetry logging to API functions.
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        # Use global variables to track state
        global _is_api_running, _pass_params  # noqa: PLW0603
        # Check if an API call is already running
        is_nested = _is_api_running
        # Check if parameters should be passed to telemetry
        pass_params = _pass_params
        # Reset global variables
        _pass_params = False
        _is_api_running = True
        try:
            # Start telemetry event scope
            with telemetry.event_scope("api", f.__name__) as event:
                try:
                    return f(*args, **kwargs)
                except Exception as exc:
                    event.error = exc.__class__.__name__
                    raise
                finally:
                    if not is_nested:
                        # Send telemetry event if not nested
                        telemetry.send_event(
                            event.interface, event.action, event.error, **event.kwargs
                        )

        finally:
            if pass_params:
                # Log parameters if pass_params flag is set
                for key, value in event.kwargs.items():
                    telemetry.log_param(key, value)
            _is_api_running = is_nested  # Restore previous API running state
            _pass_params = pass_params  # Restore previous pass_params state

    return wrapper
