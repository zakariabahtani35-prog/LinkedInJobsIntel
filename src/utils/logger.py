import logging
import sys
import io
from pathlib import Path

def get_logger(name: str, log_file: str = "reports/pipeline.log") -> logging.Logger:
    """Configures a professional logger with dual output for the terminal and files."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # Create directories if needed
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 1. Console Handler (Safe for CP1252/Windows basic terminals)
        # Using sys.stderr or stdout but forcing utf8 if supported by the OS
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 2. File Handler (UTF-8 encoding enforced)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
