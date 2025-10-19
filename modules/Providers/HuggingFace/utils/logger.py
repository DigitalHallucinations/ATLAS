# modules/Providers/HuggingFace/utils/logger.py

import logging
import os


def setup_logger(name: str = __name__, log_file: str = 'debug.log', level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if os.environ.get("PYTEST_CURRENT_TEST"):
        # During tests we let logs propagate to the root logger so fixtures such as
        # ``caplog`` can capture them reliably without interacting with file handles.
        logger.handlers = []
        logger.propagate = True
        return logger

    # Avoid adding multiple handlers if logger already has handlers
    if not logger.handlers:
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(log_file)
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.DEBUG)

        # Create formatters and add them to handlers
        c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger
