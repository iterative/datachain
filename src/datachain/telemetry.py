import logging
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

_telemetry_sent = False


def send_telemetry_once(action: str, **kwargs):
    global _telemetry_sent  # noqa: PLW0603
    if not _telemetry_sent:
        telemetry.send_event("api", action, **kwargs)
        _telemetry_sent = True
