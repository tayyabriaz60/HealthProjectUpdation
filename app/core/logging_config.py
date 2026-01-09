import logging
import sys
from pathlib import Path

# Setup logging directory
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

def setup_logging():
    """Configure structured logging for the application."""
    
    # Define log format
    log_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. Console Handler (for real-time monitoring)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)

    # 2. File Handler (for persistence)
    file_handler = logging.FileHandler(LOG_DIR / "app.log")
    file_handler.setFormatter(log_format)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Set specific levels for libraries to reduce noise
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    
    # Application Logger
    logger = logging.getLogger("healthstake")
    logger.info("Logging initialized successfully.")
    
    return logger

# Create global logger instance
logger = setup_logging()
