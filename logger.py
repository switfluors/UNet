import logging
import sys
import os
import config
from utils import get_base_folder_name

# Define the log file name
LOG_FILE = f"log.txt"

# Define a function to set up and return a logger
def get_logger(name="main"):
    logger = logging.getLogger(name)

    # Prevent adding multiple handlers in case of multiple imports
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S %p"
        ))

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Redirect print statements to logger
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)  # Redirect stderr as well

    return logger

# Helper class to redirect stdout to the logger
class StreamToLogger:
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level
        self.line_buffer = ""

    def write(self, message):
        if message.strip():  # Avoid logging empty lines
            self.logger.log(self.log_level, message.strip())

    def flush(self):
        pass  # No need to flush manually; logging handles it