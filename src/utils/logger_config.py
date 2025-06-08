# cras_project/cras_core/utils/logger_config.py
import logging
import sys

def setup_logger(logger_name, level=logging.INFO, log_to_file=False, log_file='cras_app.log'):
    """
    Sets up a logger with specified level and handlers.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False # Prevents log messages from being passed to the root logger

    # Remove existing handlers to avoid duplicate logs if this function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
    )

    # Console Handler
    ch = logging.StreamHandler(sys.stdout) # Changed from stderr to stdout for general logs
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler (Optional)
    if log_to_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger