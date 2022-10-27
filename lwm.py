import logging
import sys

from core import Core
from settings import Settings


def execute(settings_path: str, log_path: str = None) -> int:
    """
    This is where the application starts when wm is being executed.

    Args:
        app_path (str): The provided path of the application.
        args (Tuple[str]): Optional arguments from the command line.
        kwargs (Dict[str, str]): Optional keyword-arguments from the command line,
            like a configuration or log file name ``settings=xxx.yaml``.
    """
    logger = logging.getLogger("lwm")
    logger.setLevel(logging.DEBUG)

    log_handler = logging.StreamHandler(sys.stdout) \
        if log_path is None else \
        logging.FileHandler(log_path)
    log_format = "%(asctime)s [%(levelname)8s]: %(message)s (%(pathname)s:%(lineno)s)"
    log_formatter = logging.Formatter(log_format)

    log_handler.setFormatter(log_formatter)

    logger.addHandler(log_handler)

    try:
        logger.info("WM gestart.")

        settings = Settings(settings_path)

        core = Core(settings)
        core.execute()

        logger.info("WM gestopt.")

        return 0

    except Exception as e:
        logger.exception("WM afgebroken.")
        try:
            return e.errno
        except Exception:
            return 999

    finally:
        logger.removeHandler(log_handler)
