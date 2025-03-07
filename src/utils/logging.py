# utils/logging.py

"""Logging configuration for the legal assistant."""

import logging
import os
from datetime import datetime

def setup_logging(log_level=logging.INFO, log_dir="logs"):
    """Configure logging for the application.
    
    Args:
        log_level: Logging level (default: INFO)
        log_dir: Directory for log files (default: logs)
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"legal_assistant_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Log startup information
    logging.info(f"Logging configured. Log file: {log_file}")