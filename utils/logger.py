import logging
import os
from datetime import datetime
from app.config import Config

def setup_logger():
    """Setup comprehensive logging for MedGuard AI"""
    # Ensure log directory exists
    log_dir = os.path.dirname(Config.LOG_PATH)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('medguard')
    logger.setLevel(logging.INFO)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(Config.LOG_PATH)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Global logger instance
medguard_logger = setup_logger()