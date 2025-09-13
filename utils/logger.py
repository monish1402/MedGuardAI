import logging
import os
from app.config import Config

def setup_logger():
    log_dir = os.path.dirname(Config.LOG_PATH)
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('medguard')
    logger.setLevel(logging.INFO)
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler = logging.FileHandler(Config.LOG_PATH)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

medguard_logger = setup_logger()
